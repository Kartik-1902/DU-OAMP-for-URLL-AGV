function s_hat = mmse_sic_detector(Y, H, sigma2, mod_order)
% MMSE_SIC_DETECTOR  MMSE with Successive Interference Cancellation for
%   MIMO detection (per-subcarrier processing).
%
%   Inputs:
%     Y         — Received signal, size [N_rx, N_sub]
%     H         — Channel matrices, size [N_rx, N_tx, N_sub]
%     sigma2    — Noise variance (scalar)
%     mod_order — Modulation order (4=QPSK, 16=16-QAM)
%
%   Outputs:
%     s_hat — Detected symbols, size [N_tx, N_sub]
%
%   Algorithm per subcarrier:
%     1. Compute MMSE filter for all streams
%     2. Detect stream with highest post-detection SINR
%     3. Subtract its contribution (interference cancellation)
%     4. Repeat for remaining streams

    [N_rx, N_tx, N_sub] = size(H);
    constellation = get_qam_const(mod_order);  % CPU (tiny)

    % Move arrays to GPU if either input is a gpuArray
    if isa(H, 'gpuArray') || isa(Y, 'gpuArray')
        H = gpuArray(complex(H));
        Y = gpuArray(complex(Y));
        on_gpu = true;
    else
        on_gpu = false;
    end

    s_hat_gpu = zeros(N_tx, N_sub, 'like', Y);

    for k = 1:N_sub
        Hk = H(:, :, k);      % N_rx x N_tx
        yk = Y(:, k);         % N_rx x 1

        % Track which streams are detected
        remaining = 1:N_tx;
        y_residual = yk;
        H_residual = Hk;
        detected = zeros(N_tx, 1, 'like', Hk);

        for stage = 1:N_tx
            n_rem = length(remaining);

            % MMSE filter for remaining streams
            % W = (H^H * H + sigma2 * I)^{-1} * H^H
            HtH = H_residual' * H_residual;
            W_mmse = (HtH + sigma2 * eye(n_rem, 'like', H_residual)) \ (H_residual');

            % Gather filter + residual channel to CPU — at most 4×4, no GPU benefit
            % Avoids interf=0 (CPU scalar) accumulating gpuArray values
            W_cpu    = gather(W_mmse);
            H_res_cpu = gather(H_residual);
            y_res_cpu = gather(y_residual);

            % Compute post-detection SINR for each remaining stream (CPU)
            sinr = zeros(n_rem, 1);
            for s = 1:n_rem
                w_s = W_cpu(s, :).';
                desired   = abs(w_s' * H_res_cpu(:, s))^2;
                interf    = 0;
                for j = 1:n_rem
                    if j ~= s
                        interf = interf + abs(w_s' * H_res_cpu(:, j))^2;
                    end
                end
                noise_pwr = sigma2 * real(w_s' * w_s);
                sinr(s)   = real(desired) / max(real(interf + noise_pwr), 1e-12);
            end

            % Detect stream with highest SINR
            [~, best_idx] = max(sinr);
            best_stream = remaining(best_idx);

            % Hard decision (CPU — constellation is CPU)
            w_best    = W_cpu(best_idx, :).';
            x_est_cpu = w_best' * y_res_cpu;
            [~, min_idx] = min(abs(x_est_cpu - constellation));
            detected(best_stream) = constellation(min_idx);

            % Interference cancellation (GPU if H_residual is gpuArray)
            y_residual = y_residual - H_residual(:, best_idx) * detected(best_stream);

            % Remove detected stream from consideration
            H_residual(:, best_idx) = [];
            remaining(best_idx) = [];
        end

        s_hat_gpu(:, k) = detected;
    end

    % Return CPU array — callers are GPU-agnostic
    s_hat = gather(s_hat_gpu);
end


function constellation = get_qam_const(mod_order)
% Returns normalized QAM constellation points.
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
