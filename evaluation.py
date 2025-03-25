# coding: utf-8
import os
#os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import torch
import torch.nn as nn
import utils
device = torch.device("cpu")
import parameters
#device = torch.device("cuda:0")
source_domain = parameters.Global.source_domain
num_mini_batches = 74
output_path = f"./tmp/models/cyclegan/nada/saved_models"
with open(f"{output_path}/history.json", "r") as f:
    history = json.load(f)

for key in ("seg_loss", "seg_f1"):
    history[key] = utils.reduce_iterations_to_epochs(history[key], num_mini_batches)
    
n = 5

fig, axes = plt.subplots(2, 2, figsize=(14,14))
for i, prefix in enumerate(("seg_", "val_")):
    for j, key in enumerate(("loss", "f1",)):
        if i == 0:
            key = f"{prefix}{key}"
        else:
            key = f"{prefix}{key}" + f"_{source_domain}"
        axes[i, j].plot(history[key])
        axes[i, j].set_title(key)
        if i < 1:
            continue
        y = numpy.convolve(history[key], numpy.ones((n,))/n, mode="valid")
        x = range(n//2, n//2 + y.shape[0])
        axes[i, j].plot(x, y)
fig.show()

print(history["seg_f1"][-1], history[f"val_f1_{source_domain}"][-1])
print(numpy.max(history["seg_f1"]), numpy.max(history[f"val_f1_{source_domain}"]))
print(numpy.argmax(history["seg_f1"]), numpy.argmax(history[f"val_f1_{source_domain}"]), "\n")
print(sum(history[f"val_f1_{source_domain}"][-21:-1])/20)
del y
del x
del n

del history