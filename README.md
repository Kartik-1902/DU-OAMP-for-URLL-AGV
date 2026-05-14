# Deep Unfolding OAMP Detector for Ultra-Reliable AGV Control

This repository contains the final MATLAB implementation of a Deep Unfolding Orthogonal Approximate Message Passing (DU-OAMP) detector. It is designed to recover IEEE 802.11ax MIMO-OFDM control signals for Automated Guided Vehicles (AGVs) operating in highly dispersive smart warehouse environments.

## Project Overview

AGVs require Ultra-Reliable Low-Latency Communication (URLLC) with sub-1ms latency for real-time motion control. However, metallic shelving in warehouses creates severe multipath fading (up to 20 resolvable taps), causing classical linear detectors like MMSE-SIC to fail catastrophically due to error propagation.

While purely data-driven Deep Neural Networks (DNNs) can theoretically solve this, they require massive datasets, possess hundreds of thousands of parameters, and generalize poorly to unseen channel states.

Our solution is **Physics-Informed Machine Learning**: We take the iterative OAMP algorithm and "unfold" it into a neural network. Instead of learning the physics from scratch, the network only learns the optimal scalar coefficients for the iterative steps.

## Key Features

- **Tied-Parameter Architecture:** All unfolding layers share a single set of 3 learnable parameters (`gamma`, `delta`, `theta`). This reduces the optimization search space by orders of magnitude compared to per-layer learning or black-box DNNs.
- **Massive Training Efficiency:** Converges in just 5 epochs using central-difference numerical gradients (only 280 forward passes total required).
- **GPU Acceleration:** Built heavily on MATLAB's Parallel Computing Toolbox. Large matrix inversions (LMMSE filter computations) are batched on the GPU, while sequential non-linear denoisers remain on the CPU to prevent PCIe transfer bottlenecks.
- **Warehouse Channel Model:** Simulates custom multi-tap delay profiles representing metallic clutter and reflections, appended to the IEEE TGax Model D.

## Repository Structure

- `main.m` - The primary execution script. Orchestrates training, BER simulation, clutter testing, and latency benchmarking.
- `params.m` - The central configuration file containing all system, channel, and training hyperparameters.
- `deep_unfolding_oamp_train.m` - Trains the tied-parameter unfolded network using Adam optimization and numerical gradients.
- `deep_unfolding_oamp_detect.m` - The high-performance inference engine for the DU-OAMP detector.
- `blackbox_dnn_train.m` - The baseline purely data-driven DNN for comparison.
- `oamp_classical.m` - The baseline traditional OAMP iterative detector.
- `mmse_sic_detector.m` - The baseline traditional linear detector.
- `generate_channel.m` / `ofdm_modulate.m` / `ofdm_demodulate.m` - PHY layer utilities.
- `paper/` - Contains the LaTeX manuscript `main.tex` detailing the academic findings.

## Performance Highlights

1.  **Efficiency:** The DU-OAMP model requires only **3 parameters**, compared to **50,432** for the Black-Box DNN.
2.  **Reliability:** Achieves BER of $10^{-3}$ at ~20 dB SNR for QPSK, whereas traditional MMSE-SIC hits an error floor near 0.5 and fails completely.
3.  **Latency:** Achieves sub-millisecond inference for 1 and 2 layers, and an optimal accuracy-latency tradeoff at $K=3$ layers with ~1.33ms latency on GPU hardware.

## How to Run

1.  Ensure you have MATLAB R2024b with the **Parallel Computing Toolbox** and a compatible NVIDIA GPU (configured for `gpuDevice(3)` in `params.m` by default).
2.  Open the workspace and run `main.m`.
3.  The script will automatically execute all 5 phases:
    - Phase 1: Train DU-OAMP (QPSK & 16-QAM) and DNN.
    - Phase 2: Simulate BER across 0-30 dB SNR.
    - Phase 3: Test robustness against varying clutter levels.
    - Phase 4: Analyze inference latency vs unfolding depth.
    - Phase 5: Generate and save all plots to the `plots/` directory.

**NOTE**: To get PDP plot you have to run plot_pdp.m seperately, because it is not a part of main.m script.
