import torch

from body_diode_hybrid import load_ibd_pkl, predict_ibd, extract_ibd_ann_parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE = "Ibd_NN.pkl"

model, norm, checkpoint = load_ibd_pkl(MODEL_FILE, device=device)

print("Model successfully loaded.")
print(model)
print("Input names =", checkpoint["input_names"])

Vgs_test = [-5.0, -2.0, 0.0]
Vds_test = [-2.0, -2.0, -2.0]
T_C_test = [25.0, 25.0, 25.0]

Ibd_test = predict_ibd(
    model,
    norm,
    Vgs_test,
    Vds_test,
    T_C_test,
    device=device,
)

for Vgs, Vds, T_C, Ibd in zip(Vgs_test, Vds_test, T_C_test, Ibd_test.cpu().numpy().reshape(-1)):
    print(f"Vgs = {Vgs:8.3f} V, Vds = {Vds:8.3f} V, Tj = {T_C:7.2f} C, Ibd = {Ibd:12.6f} A")

ann = extract_ibd_ann_parameters(checkpoint)
for name, value in ann.items():
    print(f"{name}: shape = {value.shape}")
