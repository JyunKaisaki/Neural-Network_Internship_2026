# Surface-Potential PINN Fit Fix

Replace the corresponding files in `.github/src` and rerun `PINN_phis.ipynb`.

The revised notebook keeps the paper 2-8-1 Tanh architecture, trains one independent PINN per measured temperature, uses float64, normalizes network inputs, uses deterministic collocation grids, and verifies the trained PINN against a numerical solution that is not used as a training label.

Generated checkpoints are `phis_PINN_Tm55.pkl`, `phis_PINN_T25.pkl`, `phis_PINN_T150.pkl`, and the compatibility alias `phis_PINN.pkl` for 25 C.

The paper-style plot uses only Vds = 0, 5, 10, and 15 V. Vds = 20 V is checked separately because it is outside the plotted set in Fig. 3(b).
