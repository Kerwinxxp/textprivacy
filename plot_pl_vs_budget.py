#!/usr/bin/env python3
import re
from pathlib import Path
import matplotlib.pyplot as plt

infile = Path("posterior_leakage_results/all_new.txt")
text = infile.read_text(encoding="utf-8")
lines = text.splitlines()

pat_budget = re.compile(r"Prior Dataset: .*budget_([0-9]+(?:\.[0-9]+)?)_independent")
data = {}
for i, line in enumerate(lines):
    m = pat_budget.search(line)
    if not m:
        continue
    budget = float(m.group(1))
    mean = std = median = None
    for j in range(i+1, min(i+40, len(lines))):
        s = lines[j].strip()
        if s.startswith("Mean PL:"):
            mean = float(s.split(":",1)[1].strip())
        elif s.startswith("Std PL:"):
            std = float(s.split(":",1)[1].strip())
        elif s.startswith("Median PL:"):
            median = float(s.split(":",1)[1].strip())
        if mean is not None and std is not None and median is not None:
            break
    if mean is not None:
        data[budget] = (mean, std, median)

if not data:
    raise SystemExit("No PL data parsed from file.")

budgets = sorted(data.keys())
means = [data[b][0] for b in budgets]
stds  = [data[b][1] for b in budgets]
meds  = [data[b][2] for b in budgets]

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7,9))
axs[0].plot(budgets, means, marker='o', color='C0')
axs[0].set_ylabel("Mean PL")
axs[0].grid(True)

axs[1].plot(budgets, stds, marker='o', color='C1')
axs[1].set_ylabel("Std PL")
axs[1].grid(True)

axs[2].plot(budgets, meds, marker='o', color='C2')
axs[2].set_ylabel("Median PL")
axs[2].set_xlabel("Budget")
axs[2].grid(True)

fig.suptitle("PL statistics vs Budget")
plt.tight_layout(rect=[0,0,1,0.96])

out_path = Path("posterior_leakage_results/pl_vs_budget.png")
fig.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")