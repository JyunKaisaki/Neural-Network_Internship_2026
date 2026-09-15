# SiC MOSFET PINN project refactor

This package keeps the original project structure and applies only the changes needed to follow the supplied paper more closely.

## Main modeling choices

- `PINN_phis.ipynb` keeps the paper surface-potential PINN architecture `2-8-1` with Tanh and inputs `Vgs, phi_f`.
- `IchModeling.ipynb` removes the black-box Ich ANN and uses the physical channel-current path based on paper Eqs. (17), (18), (20), and (21).
- Output and transfer I-V datasets are used for AD-based physical-parameter extraction and joint surface-PINN fine-tuning.
- `PINN_CV.ipynb` uses the updated low-voltage and 0-1000 V capacitance datasets and the physical Cgd, Cds, and Cgs equations.
- `NN_Ibd.ipynb` keeps two hidden layers with six neurons and adds `Tj` as a third input because the paper explicitly states that temperature dependence can be introduced by adding `Tj` to the body-diode NN inputs.
- Neural networks are stored as structured PyTorch `.pkl` checkpoints instead of pickling whole Python objects.

## Dataset layout

```text
src/
├─ Template/
│  ├─ CV_Vds0-200V/
│  │  ├─ CissLowV.csv
│  │  ├─ CossLowV.csv
│  │  └─ CrssLowV.csv
│  ├─ CV_Vds0-1000V/
│  │  ├─ CissHighV.csv
│  │  ├─ CossHighV.csv
│  │  └─ CrssHighV.csv
│  ├─ Ibd-55/
│  ├─ Ibd25/
│  ├─ Ibd150/
│  ├─ OutputChara-55/
│  ├─ OutputChara25/
│  ├─ OutputChara150/
│  ├─ TransferChara-55/
│  ├─ TransferChara25/
│  └─ TransferChara150/
```

### C-V CSV format

- Column 1: `Vds` in V.
- Column 2: capacitance in pF.
- `LowV` files cover approximately 0-200 V.
- `HighV` files in the updated dataset cover approximately 0-1000 V.

### Body-diode CSV format

- Column 1: `Vds` in V.
- Column 2: `Ids` in A.
- The gate voltage is read from the filename, for example `Vgs_-5V.csv`.
- The junction temperature is read from the folder name.

### Output-characteristic CSV format

- Column 1: `Vds` in V.
- Column 2: `Ids` in A.
- The gate voltage is read from the filename.
- The junction temperature is read from the folder name.

### Transfer-characteristic CSV format

- Column 1: `Vgs` in V.
- Column 2: `Ids` in A.
- `Vds` is fixed at 20 V.
- The junction temperature is read from the folder name.

## Recommended run order

1. Run `PINN_phis.ipynb` to regenerate `phis_PINN.pkl`.
2. Run `IchModeling.ipynb` to generate `phis_PINN_IV_refined.pkl`, `Ich_physical.pkl`, and `Ich_Model.pkl`.
3. Run `PINN_CV.ipynb` to generate `phis_Cgd_PINN_refined.pkl`, `CV_physical.pkl`, and `CV_Model.pkl`.
4. Run `NN_Ibd.ipynb` to generate `Ibd_NN.pkl`.
5. Run `ModelBundle.py` to generate `SiC_MOSFET_Model.pkl`.

## Packaged neural-network checkpoints

- `phis_PINN.pkl`: pretrained surface-potential PINN.
- `phis_PINN_IV_refined.pkl`: surface-potential PINN refined with I-V data.
- `phis_Cgd_PINN.pkl`: pretrained Cgd surface-potential PINN.
- `phis_Cgd_PINN_refined.pkl`: Cgd PINN refined with capacitance data.
- `Ibd_NN.pkl`: temperature-aware body-diode ANN.
- `NeuralNetworks.pkl`: one bundle containing all neural-network checkpoints.
- `SiC_MOSFET_Model.pkl`: one bundle containing available neural-network and physical-model checkpoints.

## Important implementation boundary

The supplied paper states that the complete low-field mobility is assembled from detailed `mu_B`, `mu_SP`, `mu_C`, `mu_SR`, and `mu_DR` expressions taken from external references [5] and [28]. Those full equations are not contained in the supplied paper or source files. This package therefore uses a positive temperature-dependent effective `mu_LF(T)` polynomial and the paper velocity-saturation relation instead of inventing the missing reference equations.

The delivery `.pkl` files are executable checkpoints generated from the supplied datasets and current code. Re-running the notebooks is recommended because the notebooks perform the full configured optimization and overwrite the delivery checkpoints with the locally trained results.
