import numpy as np
from pathlib import Path

path = Path("data/processed/v1_2_label_fix/seir_v1_2_h30_seed42.npz")

data = np.load(path)

print("keys:", data.files)

for k in data.files:
    print(k, data[k].shape)

print("risk ratio:", float(np.mean(data["y_risk"])))
print("mean critical_time:", float(np.mean(data["critical_time"])))
print("min critical_time:", int(np.min(data["critical_time"])))
print("max critical_time:", int(np.max(data["critical_time"])))
