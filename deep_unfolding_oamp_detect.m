function s_hat = deep_unfolding_oamp_detect(Y, H, sigma2, mod_order, trained)
% DEEP_UNFOLDING_OAMP_DETECT  Inference using trained unfolded OAMP.
%   GPU-accelerated: Y and H are moved to GPU automatically if gpuArray.
%
%   Inputs:
%     Y       — Received signal [N_rx, N_sub]
%     H       — Channel matrices [N_rx, N_tx, N_sub]
%     sigma2  — Noise variance (scalar)
%     mod_order — Modulation order (4=QPSK, 16=16-QAM)
%     trained — Struct with learned {gamma, delta, theta, K}
%
%   Outputs:
%     s_hat — Detected symbols [N_tx, N_sub] (CPU double)

    [N_rx, N_tx, N_sub] = size(H);
    K = trained.K;

    % ---------------------------------------------------------------
    % TIED vs PER-LAYER parameter selection
    %   If trained.tied == true, all layers use the same scalar values.
    %   Otherwise fall back to per-layer vector indexing (backward compat).
    % ---------------------------------------------------------------
    if isfield(trained, 'tied') && trained.tied
        % Tied-parameter mode: single scalar shared across all layers
        gamma_s = trained.gamma_shared;
        delta_s = trained.delta_shared;
        theta_s = trained.theta_shared;
        use_tied = true;
    else
        % Per-layer mode (legacy checkpoints)
        gamma = trained.gamma;
        delta = trained.delta;
        theta = trained.theta;
        use_tied = false;
    end

    % Move large arrays to GPU for LMMSE solve; constellation stays CPU
    if isa(H, 'gpuArray') || isa(Y, 'gpuArray')
        H = gpuArray(complex(H));
        Y = gpuArray(complex(Y));
    end

    constellation = get_detect_constellation(mod_order);  % CPU
    s_hat = zeros(N_tx, N_sub);                           % CPU

    for k = 1:N_sub
        Hk = H(:, :, k);
        yk = Y(:, k);

        % LMMSE filter — runs on GPU when Hk is gpuArray
        HtH = Hk' * Hk;
        W   = (HtH + sigma2 * eye(N_tx, 'like', Hk)) \ (Hk');

        % Noise variance — gather scalar to CPU
        v_r = gather(sigma2 * real(trace(W * W')) / N_tx);
        v_r = max(v_r, 1e-10);

        % Gather W, Hk, yk to CPU once — denoiser is sequential, no GPU benefit
        Wc  = gather(W);
        Hkc = gather(Hk);
        ykc = gather(yk);
        s   = zeros(N_tx, 1);  % CPU

        for layer = 1:K
            residual = ykc - Hkc * s;

            if use_tied
                % TIED: every layer uses the same shared scalars
                r     = s + gamma_s * (Wc * residual);
                tau   = delta_s * v_r;
                s_new = mmse_denoise(r, tau, constellation);
                s     = theta_s * s_new + (1 - theta_s) * s;
            else
                % PER-LAYER: index into per-layer vectors
                r     = s + gamma(layer) * (Wc * residual);
                tau   = delta(layer) * v_r;
                s_new = mmse_denoise(r, tau, constellation);
                s     = theta(layer) * s_new + (1 - theta(layer)) * s;
            end
        end

        s_hat(:, k) = s;
    end
end


function s_out = mmse_denoise(r, tau, constellation)
% MMSE denoiser: compute E[x|r] assuming x from constellation.
    N = length(r);
    s_out = zeros(N, 1);

    for n = 1:N
        dist2 = abs(r(n) - constellation).^2;
        log_lik = -dist2 / max(tau, 1e-12);
        log_lik = log_lik - max(log_lik);
        probs = exp(log_lik);
        probs = probs / sum(probs);
        s_out(n) = sum(probs .* constellation);
    end
end


function constellation = get_detect_constellation(mod_order)
% Returns normalized QAM constellation (natural ordering).
    switch mod_order
        case 4
            constellation = (1/sqrt(2)) * [1+1j, 1-1j, -1+1j, -1-1j];
        case 16
            pam = [-3, -1, 1, 3];
            constellation = zeros(1, 16);
            idx = 1;
            for q = 1:4
                for ii = 1:4
                    constellation(idx) = pam(ii) + 1j * pam(q);
                    idx = idx + 1;
                end
            end
            constellation = constellation / sqrt(mean(abs(constellation).^2));
        otherwise
            error('Unsupported mod order: %d', mod_order);
    end
end
