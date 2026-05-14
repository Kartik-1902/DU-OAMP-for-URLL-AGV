function [trained, loss_history] = deep_unfolding_oamp_train(p, mod_order)
% DEEP_UNFOLDING_OAMP_TRAIN  Train the unfolded OAMP detector using
%   TIED-PARAMETER deep unfolding.
%
%   TIED-PARAMETER DESIGN:
%     Instead of learning separate {gamma(k), delta(k), theta(k)} for
%     each of the K unfolding layers (3*K=15 parameters for K=5),
%     all layers SHARE a single set {gamma_shared, delta_shared, theta_shared}.
%
%     This reduces the parameter count from 3*K to 3 scalars.
%
%     Impact on numerical gradient cost:
%       - Original:  2*(3*K)+1 = 31 forward passes per batch  (K=5)
%       - Tied:      2*3+1     =  7 forward passes per batch
%       - Reduction: 4.4x fewer forward passes per gradient step
%
%     Academic validity:
%       - Tied weights are standard in LISTA (Gregor & LeCun, 2010),
%         LISTA-CPSS (Chen et al., 2018), and AMP unfolding literature.
%       - Borgerding & Schniter (ISTA-Net, 2017) showed tied unfolded
%         networks achieve near-optimal performance in AWGN channels.
%       - For our 4x4 MIMO-OFDM setting, 3 tied scalars are sufficient
%         to capture the dominant gain of learned step sizes vs fixed ones.
%
%   GPU-accelerated via Parallel Computing Toolbox (gpuArray).
%
%   Inputs:
%     p         — Parameter struct from params()
%     mod_order — Modulation order (4=QPSK, 16=16-QAM)
%
%   Outputs:
%     trained      — Struct with learned {gamma_shared, delta_shared,
%                    theta_shared, K} — all CPU scalars
%     loss_history — Training loss per epoch (CPU vector)

    fprintf('=== Training Tied-Parameter DU-OAMP (mod_order=%d) ===\n', mod_order);
    fprintf('  Architecture: K=%d layers, 3 SHARED scalars (tied-parameter)\n', p.K_layers);
    fprintf('  Gradient cost: %d forward passes/batch (vs %d for per-layer)\n', ...
            2*3+1, 2*(3*p.K_layers)+1);

    use_gpu = p.use_gpu;
    K       = p.K_layers;

    % ---------------------------------------------------------------
    % 3 learnable scalars: [gamma_shared, delta_shared, theta_shared]
    % All layers use these same values at every iteration.
    % ---------------------------------------------------------------
    params_vec = [p.gamma_init, p.delta_init, p.theta_init];  % length = 3
    n_params   = 3;  % fixed — independent of K

    % Adam optimizer state (CPU — 3 scalars only)
    m      = zeros(1, n_params);
    v      = zeros(1, n_params);
    t_adam = 0;

    n_epochs   = p.n_epochs;
    batch_size = p.batch_size;
    lr         = p.learning_rate;
    eps_grad   = p.grad_epsilon;
    n_batches  = max(1, floor(p.n_train_frames / batch_size));

    fprintf('  Params: %d | Epochs: %d | Batches/epoch: %d | Batch size: %d\n', ...
            n_params, n_epochs, n_batches, batch_size);
    fprintf('  Total forward passes: %d (was ~%d with per-layer params)\n\n', ...
            n_epochs * n_batches * (2*n_params+1), ...
            n_epochs * max(1, floor(5000/64)) * (2*(3*5)+1));

    loss_history = zeros(n_epochs, 1);
    sigma2_train = 10^(-p.train_snr / 10);

    for epoch = 1:n_epochs
        % Learning rate schedule
        if epoch > p.lr_decay_epoch
            lr_cur = lr * p.lr_decay_factor;
        else
            lr_cur = lr;
        end

        epoch_loss = 0;

        for batch = 1:n_batches
            % Generate batch on CPU then move to GPU
            [Y_batch, H_batch, X_batch] = generate_train_batch( ...
                p, batch_size, mod_order, sigma2_train, use_gpu);

            % ----------------------------------------------------------
            % Central-difference numerical gradient — only 3 parameters
            % Total forward passes per batch = 2*3 + 1 = 7
            % ----------------------------------------------------------
            loss_center = gather(compute_batch_loss_tied(params_vec, K, ...
                Y_batch, H_batch, X_batch, sigma2_train, mod_order));

            grad = zeros(1, n_params);
            for i = 1:n_params
                p_plus       = params_vec;
                p_plus(i)    = p_plus(i) + eps_grad;
                loss_plus    = gather(compute_batch_loss_tied(p_plus, K, ...
                    Y_batch, H_batch, X_batch, sigma2_train, mod_order));

                p_minus      = params_vec;
                p_minus(i)   = p_minus(i) - eps_grad;
                loss_minus   = gather(compute_batch_loss_tied(p_minus, K, ...
                    Y_batch, H_batch, X_batch, sigma2_train, mod_order));

                grad(i) = (loss_plus - loss_minus) / (2 * eps_grad);
            end

            % Adam update (3-element vectors — negligible cost)
            t_adam = t_adam + 1;
            m = p.adam_beta1 * m + (1 - p.adam_beta1) * grad;
            v = p.adam_beta2 * v + (1 - p.adam_beta2) * grad.^2;
            m_hat = m / (1 - p.adam_beta1^t_adam);
            v_hat = v / (1 - p.adam_beta2^t_adam);
            params_vec = params_vec - lr_cur * m_hat ./ (sqrt(v_hat) + p.adam_eps);

            % Clamp shared parameters to physically valid ranges
            params_vec(1) = max(params_vec(1), 0.01);              % gamma > 0
            params_vec(2) = max(params_vec(2), 0.001);             % delta > 0
            params_vec(3) = max(min(params_vec(3), 0.99), 0.01);   % 0 < theta < 1

            epoch_loss = epoch_loss + loss_center;
        end

        loss_history(epoch) = epoch_loss / n_batches;
        fprintf('Epoch %d/%d | Loss: %.6f | LR: %.6f | [g=%.3f d=%.3f t=%.3f]\n', ...
                epoch, n_epochs, loss_history(epoch), lr_cur, ...
                params_vec(1), params_vec(2), params_vec(3));
    end

    % ---------------------------------------------------------------
    % Pack output — store as gamma_shared/delta_shared/theta_shared
    % (scalar fields, not vectors) so detect function can distinguish
    % tied vs per-layer architecture.
    % ---------------------------------------------------------------
    trained.gamma_shared = params_vec(1);
    trained.delta_shared = params_vec(2);
    trained.theta_shared = params_vec(3);
    trained.K            = K;
    trained.tied         = true;   % flag used by detect to select code path

    % For backward-compat with any code that reads trained.gamma as vector:
    % expand to K-element constant vectors
    trained.gamma = trained.gamma_shared * ones(1, K);
    trained.delta = trained.delta_shared * ones(1, K);
    trained.theta = trained.theta_shared * ones(1, K);

    fprintf('\nTraining complete.\n');
    fprintf('  gamma_shared = %.4f\n', trained.gamma_shared);
    fprintf('  delta_shared = %.4f\n', trained.delta_shared);
    fprintf('  theta_shared = %.4f\n', trained.theta_shared);
end


function loss = compute_batch_loss_tied(params_vec, K, Y_batch, H_batch, X_batch, sigma2, mod_order)
% COMPUTE_BATCH_LOSS_TIED  MSE loss for TIED-parameter unfolded OAMP.
%
%   params_vec = [gamma_shared, delta_shared, theta_shared]  (length=3)
%   All K layers use the SAME scalar values — no per-layer indexing.

    gamma_s = params_vec(1);   % shared step size
    delta_s = params_vec(2);   % shared noise scaling
    theta_s = params_vec(3);   % shared damping

    n_samples = size(Y_batch, 3);
    N_tx      = size(X_batch, 1);
    N_sub     = size(X_batch, 2);

    % Accumulator on same device as batch data
    if isa(Y_batch, 'gpuArray')
        total_mse = gpuArray(0);
    else
        total_mse = 0;
    end

    for b = 1:n_samples
        Y = Y_batch(:, :, b);
        X = X_batch(:, :, b);

        for sub = 1:N_sub
            Hk = H_batch(:, :, sub, b);
            yk = Y(:, sub);
            xk = X(:, sub);

            % LMMSE filter (on GPU when Hk is gpuArray)
            HtH = Hk' * Hk;
            W   = (HtH + sigma2 * eye(N_tx, 'like', Hk)) \ (Hk');

            % Effective noise variance — gather scalar to CPU
            v_r = gather(sigma2 * real(trace(W * W')) / N_tx);
            v_r = max(v_r, 1e-10);

            % Gather to CPU for denoiser loop (tiny 4-element loop)
            Wc  = gather(W);
            Hkc = gather(Hk);
            ykc = gather(yk);
            s   = zeros(N_tx, 1);  % CPU

            % K tied layers — every layer uses gamma_s, delta_s, theta_s
            for layer = 1:K
                residual = ykc - Hkc * s;
                r        = s + gamma_s * (Wc * residual);  % shared gamma

                tau   = delta_s * v_r;                      % shared delta
                s_new = soft_threshold_denoiser(r, tau, mod_order);

                s = theta_s * s_new + (1 - theta_s) * s;   % shared theta
            end

            % MSE accumulation
            if isa(Y_batch, 'gpuArray')
                xk_cpu = gather(xk);
            else
                xk_cpu = xk;
            end
            total_mse = total_mse + sum(abs(s - xk_cpu).^2);
        end
    end

    loss = total_mse / (n_samples * N_sub * N_tx);
end


function s_out = soft_threshold_denoiser(r, tau, mod_order)
% MMSE denoiser over the QAM constellation (CPU, per-element).
    constellation = get_duoamp_constellation(mod_order);
    N     = length(r);
    s_out = zeros(N, 1);
    for n = 1:N
        dist2   = abs(r(n) - constellation).^2;
        log_lik = -dist2 / max(tau, 1e-12);
        log_lik = log_lik - max(log_lik);
        probs   = exp(log_lik);
        probs   = probs / sum(probs);
        s_out(n) = sum(probs .* constellation);
    end
end


function [Y_batch, H_batch, X_batch] = generate_train_batch(p, batch_size, mod_order, sigma2, use_gpu)
% Generate training data on CPU then optionally move to GPU.

    N_tx  = p.N_tx;
    N_rx  = p.N_rx;
    N_sub = p.N_sub;

    Y_batch = zeros(N_rx, N_sub, batch_size);
    H_batch = zeros(N_rx, N_tx, N_sub, batch_size);
    X_batch = zeros(N_tx, N_sub, batch_size);

    for b = 1:batch_size
        [H_freq, ~]       = generate_channel(p);
        [~, tx_symbols, ~] = ofdm_modulate(p, mod_order);

        H_batch(:, :, :, b) = H_freq;
        X_batch(:, :, b)    = tx_symbols;

        for k = 1:N_sub
            Hk    = H_freq(:, :, k);
            xk    = tx_symbols(:, k);
            noise = sqrt(sigma2/2) * (randn(N_rx, 1) + 1j*randn(N_rx, 1));
            Y_batch(:, k, b) = Hk * xk + noise;
        end
    end

    % Move to GPU after all CPU-side generation is done
    if use_gpu
        Y_batch = gpuArray(complex(Y_batch));
        H_batch = gpuArray(complex(H_batch));
        X_batch = gpuArray(complex(X_batch));
    end
end


function constellation = get_duoamp_constellation(mod_order)
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
