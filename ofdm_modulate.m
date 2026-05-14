function [tx_signal, tx_symbols, tx_bits] = ofdm_modulate(p, mod_order)
% OFDM_MODULATE  MIMO-OFDM modulation: bit generation, symbol mapping,
%   IFFT, and cyclic prefix insertion.
%
%   Inputs:
%     p         — Parameter struct from params()
%     mod_order — Modulation order (4 for QPSK, 16 for 16-QAM)
%
%   Outputs:
%     tx_signal  — Time-domain transmit signal, size [N_fft+CP_len, N_tx]
%                  Each column is the OFDM symbol for one Tx antenna
%     tx_symbols — Frequency-domain transmitted symbols, size [N_tx, N_sub]
%                  tx_symbols(i,k) = symbol on antenna i, subcarrier k
%     tx_bits    — Binary bit matrix, size [N_tx, N_sub * bits_per_symbol]
%
%   Signal Flow:
%     1. Generate random bits
%     2. Map to constellation (QPSK or 16-QAM with Gray coding)
%     3. Apply IFFT per antenna
%     4. Add cyclic prefix

    N_tx  = p.N_tx;
    N_sub = p.N_sub;
    N_fft = p.N_fft;
    CP    = p.CP_len;

    bits_per_sym = log2(mod_order);

    % Generate random bits
    n_bits  = N_tx * N_sub * bits_per_sym;
    tx_bits = randi([0 1], N_tx, N_sub * bits_per_sym);

    % Get constellation
    constellation = get_constellation(mod_order);

    % Map bits to symbols for each antenna
    tx_symbols = zeros(N_tx, N_sub);
    for i = 1:N_tx
        bits_i = tx_bits(i, :);
        % Reshape bits into groups of bits_per_sym
        bit_groups = reshape(bits_i, bits_per_sym, N_sub).';  % [N_sub x bits_per_sym]
        % Convert to decimal indices (Gray coded)
        indices = bi2de_gray(bit_groups, bits_per_sym) + 1;  % 1-indexed
        tx_symbols(i, :) = constellation(indices);
    end

    % OFDM modulation: IFFT + CP insertion
    tx_signal = zeros(N_fft + CP, N_tx);
    for i = 1:N_tx
        % IFFT (frequency to time domain)
        x_time = ifft(tx_symbols(i, :).', N_fft);
        % Add cyclic prefix
        tx_signal(:, i) = [x_time(end-CP+1:end); x_time];
    end
end


function constellation = get_constellation(mod_order)
% GET_CONSTELLATION  Returns normalized QAM constellation with Gray coding.
%
%   Supports QPSK (mod_order=4) and 16-QAM (mod_order=16).

    switch mod_order
        case 4  % QPSK
            % Gray coded: 00->1+1j, 01->1-1j, 10->-1+1j, 11->-1-1j
            constellation = (1/sqrt(2)) * [1+1j, 1-1j, -1+1j, -1-1j];

        case 16  % 16-QAM
            % Natural ordering: matches all detectors and demodulator
            pam4 = [-3, -1, 1, 3];
            constellation = zeros(1, 16);
            idx = 1;
            for q = 1:4
                for ii = 1:4
                    constellation(idx) = pam4(ii) + 1j * pam4(q);
                    idx = idx + 1;
                end
            end
            % Normalize average power to 1
            avg_power = mean(abs(constellation).^2);
            constellation = constellation / sqrt(avg_power);

        otherwise
            error('Unsupported modulation order: %d', mod_order);
    end
end


function idx = bi2de_gray(bit_groups, bits_per_sym)
% BI2DE_GRAY  Convert binary bit groups to decimal using standard ordering.
%   bit_groups: [N x bits_per_sym] matrix, each row is one symbol's bits
%   Returns: column vector of decimal values (0-indexed)

    n = size(bit_groups, 1);
    idx = zeros(n, 1);
    for k = 1:bits_per_sym
        idx = idx + bit_groups(:, k) * 2^(bits_per_sym - k);
    end
end
