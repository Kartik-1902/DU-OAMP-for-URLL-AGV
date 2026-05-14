function [Y, H_freq] = ofdm_demodulate(p, rx_signal, H_freq_true, snr_dB)
% OFDM_DEMODULATE  MIMO-OFDM demodulation: CP removal, FFT, and noise
%   variance computation.
%
%   Inputs:
%     p            — Parameter struct from params()
%     rx_signal    — Received time-domain signal, size [N_fft+CP_len, N_rx]
%     H_freq_true  — True freq-domain channel [N_rx, N_tx, N_sub] (for ref)
%     snr_dB       — SNR in dB (for noise variance calculation)
%
%   Outputs:
%     Y      — Frequency-domain received signal, size [N_rx, N_sub]
%              Y(j,k) = received signal at antenna j, subcarrier k
%     H_freq — Frequency-domain channel (same as input, passed through)
%
%   Signal Flow:
%     1. Remove cyclic prefix from each Rx antenna
%     2. Apply FFT to get frequency-domain signal
%     3. Per-subcarrier model: Y(:,k) = H(:,:,k) * X(:,k) + N(:,k)

    N_rx  = p.N_rx;
    N_fft = p.N_fft;
    CP    = p.CP_len;

    % Initialize output
    Y = zeros(N_rx, N_fft);

    for j = 1:N_rx
        % Remove cyclic prefix
        rx_no_cp = rx_signal(CP+1:end, j);

        % Apply FFT (time to frequency domain)
        Y(j, :) = fft(rx_no_cp, N_fft).';
    end

    % Pass through channel (caller already has it)
    H_freq = H_freq_true;
end


function H_hat = estimate_channel_ls(p, Y_pilot, X_pilot, pilot_indices)
% ESTIMATE_CHANNEL_LS  Least-Squares pilot-based channel estimation.
%
%   Inputs:
%     p             — Parameter struct
%     Y_pilot       — Received signal at pilot subcarriers [N_rx, n_pilots]
%     X_pilot       — Transmitted pilot symbols [N_tx, n_pilots]
%     pilot_indices — Indices of pilot subcarriers
%
%   Outputs:
%     H_hat — Estimated channel [N_rx, N_tx, N_sub] (interpolated)

    N_rx  = p.N_rx;
    N_tx  = p.N_tx;
    N_sub = p.N_sub;
    n_pilots = length(pilot_indices);

    % LS estimation at pilot positions
    H_pilot = zeros(N_rx, N_tx, n_pilots);
    for pp = 1:n_pilots
        % Y(:,p) = H(:,:,p) * X(:,p)
        % For orthogonal pilots: H_hat = Y * X^H * inv(X * X^H)
        yp = Y_pilot(:, pp);
        xp = X_pilot(:, pp);
        % Simple LS per pilot (assuming one Tx active at a time for pilots)
        for i = 1:N_tx
            if abs(xp(i)) > 0
                H_pilot(:, i, pp) = yp / xp(i);
            end
        end
    end

    % Linear interpolation across subcarriers
    H_hat = zeros(N_rx, N_tx, N_sub);
    sub_indices = 1:N_sub;
    for j = 1:N_rx
        for i = 1:N_tx
            h_vals = squeeze(H_pilot(j, i, :));
            H_hat(j, i, :) = interp1(pilot_indices, h_vals, ...
                              sub_indices, 'linear', 'extrap');
        end
    end
end
