%% plot_pdp.m
% Generates the Power Delay Profile (PDP) comparison figure for the paper.
% Plots TGax Model D baseline vs Low / Medium / High warehouse clutter.
% Output: plots/pdp_comparison.png

p = params();

% Ensure output directory exists (same pattern as plot_results.m)
if ~isfolder('plots'), mkdir('plots'); end

% Use same color palette as plot_results.m
colors = lines(7);

figure('Name', 'PDP Comparison', 'Position', [100 100 750 480]);
hold on;

% ---- Base TGax Model D ----
delays_base = p.tgax_delays_ns;
powers_base = p.tgax_powers_dB;
stem(delays_base, powers_base, 'Color', colors(1,:), 'LineWidth', 1.5, ...
    'MarkerFaceColor', colors(1,:), 'MarkerSize', 5, ...
    'DisplayName', 'TGax Model D (Baseline)');

% ---- Clutter scenarios ----
clutter_taps  = [2, 5, 10];
clutter_names = {'Low Clutter (2 extra taps)', ...
                  'Medium Clutter (5 extra taps)', ...
                  'High Clutter (10 extra taps)'};

rng(42);   % reproducible extra taps

for c = 1:3
    n_extra = clutter_taps(c);
    % Random extra delays between 60 and 200 ns
    extra_delays = sort(round(p.extra_delay_min + ...
        (p.extra_delay_max - p.extra_delay_min) * rand(1, n_extra)));
    % Generate n_extra power values linearly spaced between -3 and -7 dB
    extra_powers = linspace(-3, -7, n_extra);

    stem(extra_delays, extra_powers, 'Color', colors(c+1,:), 'LineWidth', 1.5, ...
        'MarkerFaceColor', colors(c+1,:), 'MarkerSize', 5, ...
        'DisplayName', clutter_names{c});
end

hold off;
xlabel('Delay (ns)', 'FontSize', 12);
ylabel('Relative Power (dB)', 'FontSize', 12);
title('Power Delay Profile: TGax Model D vs Warehouse Clutter', 'FontSize', 13);
legend('Location', 'northeast', 'FontSize', 10);
grid on;
xlim([-10, 510]);
ylim([-25, 3]);
set(gca, 'FontSize', 11);

saveas(gcf, fullfile('plots', 'pdp_comparison.png'));
fprintf('Saved: plots/pdp_comparison.png\n');
