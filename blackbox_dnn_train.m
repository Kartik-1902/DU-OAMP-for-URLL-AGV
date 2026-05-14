function [dnn, loss_history] = blackbox_dnn_train(p, mod_order)
% BLACKBOX_DNN_TRAIN  Train a fully-connected DNN detector for MIMO.
%   Implements a 4-layer FC network with manual backpropagation and Adam
%   optimizer (NO Deep Learning Toolbox required).
%
%   Architecture:
%     Input → 256 → 128 → 64 → Output
%     ReLU activations on hidden layers
%     Separate softmax per antenna on output layer
%     Cross-entropy loss
%
%   Inputs:
%     p         — Parameter struct from params()
%     mod_order — Modulation order (4=QPSK, 16=16-QAM)
%
%   Outputs:
%     dnn          — Struct with trained weights and biases
%     loss_history — Training loss per epoch

    fprintf('=== Training Black-Box DNN (mod_order=%d) ===\n', mod_order);

    N_tx = p.N_tx;
    N_rx = p.N_rx;
    use_gpu = p.use_gpu;

    % Input: [real(y); imag(y); real(H_vec); imag(H_vec)]
    input_size = 2*N_rx + 2*N_rx*N_tx;  % 8 + 32 = 40

    % Output: softmax over constellation per antenna
    n_const = mod_order;  % Number of constellation points
    output_size = N_tx * n_const;  % 4*4=16 for QPSK, 4*16=64 for 16-QAM

    % Network architecture
    layer_sizes = [input_size, p.dnn_hidden_sizes, output_size];
    n_layers = length(layer_sizes) - 1;

    % Initialize weights on CPU first (He initialization)
    W = cell(n_layers, 1);
    b = cell(n_layers, 1);
    for l = 1:n_layers
        fan_in = layer_sizes(l);
        W{l} = randn(layer_sizes(l+1), fan_in) * sqrt(2/fan_in);
        b{l} = zeros(layer_sizes(l+1), 1);
    end

    % Move weights and biases to GPU
    if use_gpu
        for l = 1:n_layers
            W{l} = gpuArray(W{l});
            b{l} = gpuArray(b{l});
        end
        fprintf('DNN weights moved to GPU.\n');
    end

    % Get constellation for label generation
    constellation = get_dnn_constellation(mod_order);

    % Adam state (same device as weights)
    mW = cell(n_layers, 1);  vW = cell(n_layers, 1);
    mb = cell(n_layers, 1);  vb = cell(n_layers, 1);
    for l = 1:n_layers
        mW{l} = zeros(size(W{l}), 'like', W{l});  vW{l} = zeros(size(W{l}), 'like', W{l});
        mb{l} = zeros(size(b{l}), 'like', b{l});  vb{l} = zeros(size(b{l}), 'like', b{l});
    end

    % Training settings
    n_epochs   = p.n_epochs;
    batch_size = p.batch_size;
    lr         = p.dnn_lr;
    n_batches  = max(1, floor(p.n_train_frames / batch_size));
    t_adam     = 0;

    loss_history = zeros(n_epochs, 1);

    for epoch = 1:n_epochs
        epoch_loss = 0;

        % Learning rate decay
        if epoch > p.lr_decay_epoch
            lr_cur = lr * p.lr_decay_factor;
        else
            lr_cur = lr;
        end

        for batch = 1:n_batches
            % Generate training data and move to GPU
            [inputs, labels] = generate_dnn_batch(p, batch_size, mod_order, ...
                                                   p.train_snr, constellation, use_gpu);
            n_samples = size(inputs, 2);

            % === Forward Pass ===
            a = cell(n_layers + 1, 1);
            z = cell(n_layers, 1);
            a{1} = inputs;  % input_size x n_samples

            for l = 1:n_layers
                z{l} = W{l} * a{l} + b{l};  % pre-activation

                if l < n_layers
                    % ReLU for hidden layers
                    a{l+1} = max(0, z{l});
                else
                    % Softmax per antenna for output layer
                    a{l+1} = apply_per_antenna_softmax(z{l}, N_tx, n_const);
                end
            end

            % === Compute Cross-Entropy Loss ===
            output = a{n_layers + 1};  % output_size x n_samples
            % Clamp for numerical stability
            output_clamped = max(output, 1e-12);
            loss = -sum(labels .* log(output_clamped), 1);
            batch_loss = mean(loss);
            epoch_loss = epoch_loss + batch_loss;

            % === Backward Pass ===
            dW = cell(n_layers, 1);
            db = cell(n_layers, 1);

            % Output layer gradient (softmax + cross-entropy)
            delta = output - labels;  % output_size x n_samples

            for l = n_layers:-1:1
                % Gradient for weights and biases
                dW{l} = (delta * a{l}') / n_samples;
                db{l} = mean(delta, 2);

                if l > 1
                    % Propagate gradient through activation
                    delta = W{l}' * delta;
                    % ReLU derivative
                    delta = delta .* (z{l-1} > 0);
                end
            end

            % === Adam Update ===
            t_adam = t_adam + 1;
            for l = 1:n_layers
                % Weights
                mW{l} = p.adam_beta1 * mW{l} + (1 - p.adam_beta1) * dW{l};
                vW{l} = p.adam_beta2 * vW{l} + (1 - p.adam_beta2) * dW{l}.^2;
                mW_hat = mW{l} / (1 - p.adam_beta1^t_adam);
                vW_hat = vW{l} / (1 - p.adam_beta2^t_adam);
                W{l} = W{l} - lr_cur * mW_hat ./ (sqrt(vW_hat) + p.adam_eps);

                % Biases
                mb{l} = p.adam_beta1 * mb{l} + (1 - p.adam_beta1) * db{l};
                vb{l} = p.adam_beta2 * vb{l} + (1 - p.adam_beta2) * db{l}.^2;
                mb_hat = mb{l} / (1 - p.adam_beta1^t_adam);
                vb_hat = vb{l} / (1 - p.adam_beta2^t_adam);
                b{l} = b{l} - lr_cur * mb_hat ./ (sqrt(vb_hat) + p.adam_eps);
            end
        end

        loss_history(epoch) = epoch_loss / n_batches;

        if mod(epoch, 10) == 0 || epoch == 1
            fprintf('Epoch %3d/%d | Loss: %.6f\n', epoch, n_epochs, ...
                    loss_history(epoch));
        end
    end

    % Pack trained network — gather weights back to CPU for saving
    dnn.W = cell(n_layers, 1);
    dnn.b = cell(n_layers, 1);
    for l = 1:n_layers
        dnn.W{l} = gather(W{l});
        dnn.b{l} = gather(b{l});
    end
    dnn.n_layers = n_layers;
    dnn.N_tx = N_tx;
    dnn.n_const = n_const;
    dnn.mod_order = mod_order;
    dnn.constellation = constellation;

    fprintf('DNN training complete. Final loss: %.6f\n', loss_history(end));
end


function s_hat = blackbox_dnn_detect(Y, H, dnn)
% BLACKBOX_DNN_DETECT  Inference using the trained black-box DNN detector.
%
%   Inputs:
%     Y   — Received signal [N_rx, N_sub]
%     H   — Channel [N_rx, N_tx, N_sub]
%     dnn — Trained DNN struct from blackbox_dnn_train
%
%   Outputs:
%     s_hat — Detected symbols [N_tx, N_sub]

    [~, N_tx, N_sub] = size(H);

    s_hat = zeros(N_tx, N_sub);

    for k = 1:N_sub
        Hk = H(:, :, k);
        yk = Y(:, k);

        % Build input vector
        input_vec = [real(yk); imag(yk); real(Hk(:)); imag(Hk(:))];

        % Forward pass
        a = input_vec;
        for l = 1:dnn.n_layers
            z = dnn.W{l} * a + dnn.b{l};
            if l < dnn.n_layers
                a = max(0, z);  % ReLU
            else
                a = apply_per_antenna_softmax(z, dnn.N_tx, dnn.n_const);
            end
        end

        % Select constellation point with highest probability per antenna
        for n = 1:N_tx
            probs = a((n-1)*dnn.n_const + 1 : n*dnn.n_const);
            [~, idx] = max(probs);
            s_hat(n, k) = dnn.constellation(idx);
        end
    end
end


function output = apply_per_antenna_softmax(z, N_tx, n_const)
% APPLY_PER_ANTENNA_SOFTMAX  Apply softmax separately for each antenna.
%   z: [N_tx*n_const, n_samples]
    n_samples = size(z, 2);
    output = zeros(size(z));
    for n = 1:N_tx
        idx = (n-1)*n_const + 1 : n*n_const;
        z_ant = z(idx, :);
        z_ant = z_ant - max(z_ant, [], 1);  % Numerical stability
        exp_z = exp(z_ant);
        output(idx, :) = exp_z ./ sum(exp_z, 1);
    end
end


function [inputs, labels] = generate_dnn_batch(p, batch_size, mod_order, ...
                                                snr_dB, constellation, use_gpu)
% GENERATE_DNN_BATCH  Generate training data for black-box DNN.
%   Creates input features and one-hot labels, then moves to GPU.

    N_tx = p.N_tx;
    N_rx = p.N_rx;
    N_sub = p.N_sub;
    n_const = mod_order;

    sigma2 = 10^(-snr_dB/10);
    input_size = 2*N_rx + 2*N_rx*N_tx;
    output_size = N_tx * n_const;

    total_samples = batch_size * N_sub;
    inputs = zeros(input_size, total_samples);
    labels = zeros(output_size, total_samples);

    sample_idx = 0;
    for b = 1:batch_size
        % Generate channel and symbols
        [H_freq, ~] = generate_channel(p);
        [~, tx_symbols, ~] = ofdm_modulate(p, mod_order);

        for k = 1:N_sub
            sample_idx = sample_idx + 1;
            Hk = H_freq(:, :, k);
            xk = tx_symbols(:, k);

            % Received signal with noise
            noise = sqrt(sigma2/2) * (randn(N_rx, 1) + 1j*randn(N_rx, 1));
            yk = Hk * xk + noise;

            % Input: [real(y); imag(y); real(H); imag(H)]
            inputs(:, sample_idx) = [real(yk); imag(yk); ...
                                     real(Hk(:)); imag(Hk(:))];

            % One-hot labels per antenna
            for n = 1:N_tx
                [~, const_idx] = min(abs(xk(n) - constellation));
                labels((n-1)*n_const + const_idx, sample_idx) = 1;
            end
        end
    end

    % Shuffle
    perm = randperm(total_samples);
    inputs = inputs(:, perm);
    labels = labels(:, perm);

    % Move to GPU — large matrices benefit from GPU matrix multiply
    if use_gpu
        inputs = gpuArray(inputs);
        labels = gpuArray(labels);
    end
end


function constellation = get_dnn_constellation(mod_order)
% Returns normalized QAM constellation.
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
