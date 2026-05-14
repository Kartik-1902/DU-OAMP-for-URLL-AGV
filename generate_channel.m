function [H_freq, h_time] = generate_channel(p, n_extra_taps)
% GENERATE_CHANNEL  Generate IEEE 802.11ax TGax Model D MIMO channel with
%   warehouse-specific multipath extensions.
%
%   Inputs:
%     p            — Parameter struct from params()
%     n_extra_taps — Number of extra warehouse taps to add (overrides p.n_extra_taps)
%
%   Outputs:
%     H_freq — Frequency-domain channel matrix, size [N_rx, N_tx, N_sub]
%              H_freq(:,:,k) is the N_rx x N_tx channel matrix at subcarrier k
%     h_time — Time-domain channel taps, cell array {N_rx, N_tx}
%              h_time{j,i} is the channel impulse response from Tx i to Rx j
%
%   Channel Model:
%     - Base: TGax Model D (dense indoor office) with ~15 taps
%     - Extension: Additional high-power taps simulating metal shelving
%       reflections in a smart warehouse environment
%     - Each tap is complex Gaussian (Rayleigh fading) with specified power

    if nargin < 2
        n_extra_taps = p.n_extra_taps;
    end

    N_tx  = p.N_tx;
    N_rx  = p.N_rx;
    N_fft = p.N_fft;
    fs    = p.fs;

    % Convert base tap delays from nanoseconds to sample indices
    sample_period_ns = 1e9 / fs;  % 50 ns for 20 MHz
    base_tap_indices = round(p.tgax_delays_ns / sample_period_ns);
    base_tap_powers  = 10.^(p.tgax_powers_dB / 10);

    % Generate extra warehouse taps (metal shelving reflections)
    if n_extra_taps > 0
        extra_delays_ns = linspace(p.extra_delay_min, p.extra_delay_max, n_extra_taps);
        extra_tap_indices = round(extra_delays_ns / sample_period_ns);

        % Power for extra taps (high power reflections from metal)
        if n_extra_taps <= length(p.extra_power_dB)
            extra_powers = 10.^(p.extra_power_dB(1:n_extra_taps) / 10);
        else
            extra_powers_dB = linspace(p.extra_power_dB(1), ...
                              p.extra_power_dB(end), n_extra_taps);
            extra_powers = 10.^(extra_powers_dB / 10);
        end
    else
        extra_tap_indices = [];
        extra_powers = [];
    end

    % Combine base and extra taps
    all_tap_indices = [base_tap_indices, extra_tap_indices];
    all_tap_powers  = [base_tap_powers, extra_powers];

    % Maximum channel length in samples
    max_delay = max(all_tap_indices);
    L = max_delay + 1;  % Channel length

    % Normalize total power to 1
    total_power = sum(all_tap_powers);
    all_tap_powers = all_tap_powers / total_power;

    % Generate channel for each Tx-Rx pair
    h_time = cell(N_rx, N_tx);
    H_freq = zeros(N_rx, N_tx, N_fft);

    for j = 1:N_rx
        for i = 1:N_tx
            % Initialize channel impulse response
            h = zeros(L, 1);

            % Generate complex Gaussian taps with specified power
            for t = 1:length(all_tap_indices)
                tap_idx = all_tap_indices(t) + 1;  % 1-indexed
                tap_power = all_tap_powers(t);
                % Complex Gaussian: CN(0, tap_power)
                h(tap_idx) = h(tap_idx) + ...
                    sqrt(tap_power/2) * (randn + 1j*randn);
            end

            h_time{j, i} = h;

            % Convert to frequency domain via FFT
            H_f = fft(h, N_fft);
            H_freq(j, i, :) = H_f;
        end
    end
end
