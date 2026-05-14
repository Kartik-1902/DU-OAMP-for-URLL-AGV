function [ber, n_errors, n_bits] = compute_ber(tx_bits, rx_bits)
% COMPUTE_BER  Compute bit error rate between transmitted and received bits.
%
%   Inputs:
%     tx_bits  — Transmitted bit vector (1D or 2D array)
%     rx_bits  — Received/detected bit vector (same size as tx_bits)
%
%   Outputs:
%     ber      — Bit error rate (scalar)
%     n_errors — Number of bit errors (scalar)
%     n_bits   — Total number of bits compared (scalar)
%
%   Example:
%     [ber, ne, nb] = compute_ber([0 1 1 0], [0 1 0 0]);
%     % ber = 0.25, ne = 1, nb = 4

    tx_bits = tx_bits(:);
    rx_bits = rx_bits(:);

    if length(tx_bits) ~= length(rx_bits)
        error('compute_ber: tx_bits and rx_bits must have the same length.');
    end

    n_bits   = length(tx_bits);
    n_errors = sum(tx_bits ~= rx_bits);
    ber      = n_errors / n_bits;
end
