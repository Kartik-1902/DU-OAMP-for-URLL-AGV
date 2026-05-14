function s_hat = oamp_classical(Y, H, sigma2, mod_order, K)
% OAMP_CLASSICAL  Classical Orthogonal AMP detector for MIMO detection.
%   Implements K iterations of OAMP with LMMSE linear estimator and
%   MMSE-optimal non-linear denoiser based on the constellation.
%
%   Inputs:
%     Y         — Received signal per subcarrier, size [N_rx, N_sub]
%     H         — Channel matrices, size [N_rx, N_tx, N_sub]
%     sigma2    — Noise variance (scalar)
%     mod_order — Modulation order (4=QPSK, 16=16-QAM)
%     K         — Number of OAMP iterations (default: 5)
%
%   Outputs:
%     s_hat — Detected symbols, size [N_tx, N_sub]
%
%   Algorithm per subcarrier k:
%     For t = 1 to K:
%       1. Linear Estimator (LMMSE):
%          W = (H^H*H + sigma2*I)^{-1} * H^H
%          r = s_prev + W * (Y - H * s_prev)
%       2. Compute effective noise variance for denoiser
%       3. Non-linear Estimator (MMSE denoiser over constellation)
%          s = E[x | r, v] where v is the residual variance

    if nargin < 5, K = 5; end

    [N_rx, N_tx, N_sub] = size(H);
    constellation = get_qam_constellation(mod_order);  % CPU (tiny)

    % Move arrays to GPU for LMMSE solve; constellation stays CPU
    if isa(H, 'gpuArray') || isa(Y, 'gpuArray')
        H = gpuArray(complex(H));
        Y = gpuArray(complex(Y));
    end

    s_hat = zeros(N_tx, N_sub);  % CPU — s gathered per subcarrier

    for k = 1:N_sub
        Hk = H(:, :, k);           % N_rx x N_tx
        yk = Y(:, k);              % N_rx x 1

        % Precompute LMMSE filter matrix — runs on GPU when Hk is gpuArray
        HtH = Hk' * Hk;
        W = (HtH + sigma2 * eye(N_tx, 'like', Hk)) \ (Hk');  % N_tx x N_rx

        % Orthogonalization factor (unused in fixed-step version, kept for reference)
        % trace_B = real(trace(W * Hk)) / N_tx;

        % Effective noise variance — gather scalar
        v_r = gather(sigma2 * real(trace(W * W')) / N_tx);
        v_r = max(v_r, 1e-10);

        % Gather W, Hk, yk to CPU once — denoiser is 4-element sequential loop
        % GPU benefit was the matrix solve above; no benefit in the K-loop
        Wc  = gather(W);
        Hkc = gather(Hk);
        ykc = gather(yk);
        s   = zeros(N_tx, 1);  % CPU

        for t = 1:K
            % === Linear Estimator (CPU) ===
            residual = ykc - Hkc * s;
            r = s + Wc * residual;

            % === Non-linear Estimator (MMSE denoiser, CPU) ===
            [s, ~] = mmse_denoiser(r, v_r, constellation);
        end

        s_hat(:, k) = s;
    end
end


function [s_out, v_out] = mmse_denoiser(r, v_r, constellation)
% MMSE_DENOISER  Compute posterior mean E[x|r] assuming x is from a finite
%   constellation and r = x + noise with noise variance v_r.
%
%   Inputs:
%     r             — Observation vector, size [N, 1]
%     v_r           — Noise variance (scalar)
%     constellation — QAM constellation points, size [1, M]
%
%   Outputs:
%     s_out — Posterior mean, size [N, 1]
%     v_out — Posterior variance (scalar, averaged)

    N = length(r);
    M = length(constellation);

    % Compute log-likelihoods for each constellation point
    % p(c | r_n) proportional to exp(-|r_n - c|^2 / v_r)
    s_out = zeros(N, 1);
    v_out_sum = 0;

    for n = 1:N
        % Distances to all constellation points
        dist2 = abs(r(n) - constellation).^2;  % 1 x M

        % Log-likelihood (subtract max for numerical stability)
        log_lik = -dist2 / v_r;
        log_lik = log_lik - max(log_lik);

        % Probabilities
        probs = exp(log_lik);
        probs = probs / sum(probs);

        % Posterior mean
        s_out(n) = sum(probs .* constellation);

        % Posterior variance
        v_out_sum = v_out_sum + sum(probs .* abs(constellation - s_out(n)).^2);
    end

    v_out = real(v_out_sum) / N;
end


function constellation = get_qam_constellation(mod_order)
% GET_QAM_CONSTELLATION  Returns QAM constellation normalized to unit power.

    switch mod_order
        case 4  % QPSK
            constellation = (1/sqrt(2)) * [1+1j, 1-1j, -1+1j, -1-1j];

        case 16  % 16-QAM
            pam = [-3, -1, 1, 3];
            constellation = zeros(1, 16);
            idx = 1;
            for q = 1:4
                for ii = 1:4
                    constellation(idx) = pam(ii) + 1j * pam(q);
                    idx = idx + 1;
                end
            end
            avg_power = mean(abs(constellation).^2);
            constellation = constellation / sqrt(avg_power);

        otherwise
            error('Unsupported modulation order: %d', mod_order);
    end
end
