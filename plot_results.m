function plot_results(results, p)
% PLOT_RESULTS  Generate all four publication-quality result plots.
%
%   Inputs:
%     results — Struct containing simulation results:
%       .ber_qpsk    — [4 x n_snr] BER for QPSK (DU-OAMP, OAMP, MMSE-SIC, DNN)
%       .ber_16qam   — [4 x n_snr] BER for 16-QAM
%       .clutter_ber — [3 x n_snr] BER for 3 clutter levels (DU-OAMP)
%       .latency     — [n_K x 1] latency per frame (ms) for each K
%       .latency_ber — [n_K x 1] BER at SNR=15dB for each K
%       .loss_duoamp — [n_epochs x 1] training loss for DU-OAMP
%       .loss_dnn    — [n_epochs x 1] training loss for DNN
%     p       — Parameter struct from params()
%
%   Generates:
%     Plot 1: BER vs SNR (main result, QPSK + 16-QAM subplots)
%     Plot 2: BER vs SNR at different warehouse clutter levels
%     Plot 3: Processing latency vs number of unfolding layers
%     Plot 4: Convergence during training

    SNR_range = p.SNR_range;
    colors = lines(7);

    % Ensure output directory exists
    if ~isfolder('plots'), mkdir('plots'); end

    %% ===== Plot 1: BER vs SNR (Main Result) =====
    figure('Name', 'Plot 1: BER vs SNR', 'Position', [50 50 1200 500]);

    % QPSK subplot
    subplot(1, 2, 1);
    semilogy(SNR_range, results.ber_qpsk(1,:), 'o-', 'Color', colors(1,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Deep Unfolding OAMP');
    hold on;
    semilogy(SNR_range, results.ber_qpsk(2,:), 's--', 'Color', colors(2,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Classical OAMP');
    semilogy(SNR_range, results.ber_qpsk(3,:), 'd-.', 'Color', colors(3,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'MMSE-SIC');
    semilogy(SNR_range, results.ber_qpsk(4,:), '^:', 'Color', colors(4,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Black-Box DNN');
    hold off;
    grid on;
    xlabel('Average SNR (dB)', 'FontSize', 12);
    ylabel('Bit Error Rate (BER)', 'FontSize', 12);
    title('QPSK — 4×4 MIMO-OFDM Warehouse Channel', 'FontSize', 13);
    legend('Location', 'southwest', 'FontSize', 10);
    ylim([1e-5 1]);
    xlim([SNR_range(1) SNR_range(end)]);
    set(gca, 'FontSize', 11);

    % 16-QAM subplot
    subplot(1, 2, 2);
    semilogy(SNR_range, results.ber_16qam(1,:), 'o-', 'Color', colors(1,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Deep Unfolding OAMP');
    hold on;
    semilogy(SNR_range, results.ber_16qam(2,:), 's--', 'Color', colors(2,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Classical OAMP');
    semilogy(SNR_range, results.ber_16qam(3,:), 'd-.', 'Color', colors(3,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'MMSE-SIC');
    semilogy(SNR_range, results.ber_16qam(4,:), '^:', 'Color', colors(4,:), ...
             'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Black-Box DNN');
    hold off;
    grid on;
    xlabel('Average SNR (dB)', 'FontSize', 12);
    ylabel('Bit Error Rate (BER)', 'FontSize', 12);
    title('16-QAM — 4×4 MIMO-OFDM Warehouse Channel', 'FontSize', 13);
    legend('Location', 'southwest', 'FontSize', 10);
    ylim([1e-5 1]);
    xlim([SNR_range(1) SNR_range(end)]);
    set(gca, 'FontSize', 11);

    sgtitle('BER vs SNR — Deep Unfolding OAMP vs Baselines', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, fullfile('plots', 'plot1_ber_vs_snr.png'));
    fprintf('Saved: plots/plot1_ber_vs_snr.png\n');

    %% ===== Plot 2: BER vs SNR at Different Clutter Levels =====
    figure('Name', 'Plot 2: Clutter Levels', 'Position', [100 100 700 500]);

    clutter_markers = {'o-', 's--', 'd-.'};
    clutter_colors  = [colors(1,:); colors(5,:); colors(6,:)];

    for c = 1:3
        semilogy(SNR_range, results.clutter_ber(c,:), clutter_markers{c}, ...
                 'Color', clutter_colors(c,:), 'LineWidth', 2, 'MarkerSize', 7, ...
                 'DisplayName', p.clutter_labels{c});
        hold on;
    end
    hold off;
    grid on;
    xlabel('Average SNR (dB)', 'FontSize', 12);
    ylabel('Bit Error Rate (BER)', 'FontSize', 12);
    title('Deep Unfolding OAMP — Warehouse Clutter Impact', 'FontSize', 13);
    legend('Location', 'southwest', 'FontSize', 11);
    ylim([1e-5 1]);
    xlim([SNR_range(1) SNR_range(end)]);
    set(gca, 'FontSize', 11);
    saveas(gcf, fullfile('plots', 'plot2_clutter_levels.png'));
    fprintf('Saved: plots/plot2_clutter_levels.png\n');

    %% ===== Plot 3: Latency vs Number of Layers =====
    figure('Name', 'Plot 3: Latency', 'Position', [150 150 700 500]);

    K_test = p.K_layers_test;

    yyaxis left;
    plot(K_test, results.latency, 'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
         'Color', colors(1,:));
    hold on;
    yline(1, '--r', 'URLLC 1ms Limit', 'LineWidth', 1.5, 'FontSize', 10, ...
          'LabelHorizontalAlignment', 'left');
    hold off;
    ylabel('Inference Latency (ms)', 'FontSize', 12);
    set(gca, 'YColor', colors(1,:));

    yyaxis right;
    plot(K_test, results.latency_ber, 's-', 'LineWidth', 2, 'MarkerSize', 8, ...
         'Color', colors(2,:));
    ylabel('BER at SNR = 15 dB', 'FontSize', 12);
    set(gca, 'YColor', colors(2,:), 'YScale', 'log');

    xlabel('Number of Unfolding Layers (K)', 'FontSize', 12);
    title('Latency-Accuracy Tradeoff vs Unfolding Depth', 'FontSize', 13);
    grid on;
    set(gca, 'FontSize', 11);
    xticks(K_test);
    legend({'Latency', 'BER'}, 'Location', 'north', 'FontSize', 11);
    saveas(gcf, fullfile('plots', 'plot3_latency_vs_layers.png'));
    fprintf('Saved: plots/plot3_latency_vs_layers.png\n');

    %% ===== Plot 4: Training Convergence =====
    figure('Name', 'Plot 4: Convergence', 'Position', [200 200 700 500]);

    semilogy(1:length(results.loss_duoamp), results.loss_duoamp, '-', ...
         'LineWidth', 2, 'Color', colors(1,:), 'DisplayName', 'Deep Unfolding OAMP (MSE)');
    hold on;
    semilogy(1:length(results.loss_dnn), results.loss_dnn, '--', ...
         'LineWidth', 2, 'Color', colors(4,:), 'DisplayName', 'Black-Box DNN (Cross-Entropy)');
    hold off;
    grid on;
    xlabel('Training Epoch', 'FontSize', 12);
    ylabel('Training Loss (log scale)', 'FontSize', 12);
    title('Training Convergence Comparison', 'FontSize', 13);
    legend('Location', 'northeast', 'FontSize', 11);
    set(gca, 'FontSize', 11);
    saveas(gcf, fullfile('plots', 'plot4_convergence.png'));
    fprintf('Saved: plots/plot4_convergence.png\n');

    fprintf('All plots generated and saved.\n');
end
