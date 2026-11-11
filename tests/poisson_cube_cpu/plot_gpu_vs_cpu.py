#!/usr/bin/env python3
"""Parse the result files of run_gpu_vs_cpu.sh and plot throughput (DoF/s).

usage: plot_gpu_vs_cpu.py <jobid> [results_dir]
reads  <results_dir>/<label>.<jobid>.out for the labels below
writes <results_dir>/gpu_vs_cpu.<jobid>.png and .csv
"""
import re
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jobid = sys.argv[1]
rdir = sys.argv[2] if len(sys.argv) > 2 else "."

RUNS = [  # label, legend, color, linestyle
    ("gpu64.np1",   "GPU bk3: 1 H100 (fp64)",                "tab:red",    "-"),
    ("gpu64.np2",   "GPU bk3: 2 H100 (fp64)",                "tab:red",    "--"),
    ("gpu64.np4",   "GPU bk3: 4 H100 = node (fp64)",         "tab:red",    ":"),
    ("cpu64.np72",  "CPU: 72 cores = 1 Grace (fp64)",        "tab:blue",   "-"),
    ("cpu64.np144", "CPU: 144 cores = 2 Grace (fp64)",       "tab:blue",   "--"),
    ("cpu64.np288", "CPU: 288 cores = node (fp64)",          "tab:blue",   ":"),
    ("gpu32.np1",   "GPU bk3: 1 H100 (fp32 V-cycle)",        "tab:orange", "-"),
    ("gpu32.np2",   "GPU bk3: 2 H100 (fp32 V-cycle)",        "tab:orange", "--"),
    ("gpu32.np4",   "GPU bk3: 4 H100 = node (fp32 V-cycle)", "tab:orange", ":"),
    ("cpu32.np72",  "CPU: 72 cores (fp32 V-cycle)",          "tab:green",  "-"),
    ("cpu32.np144", "CPU: 144 cores (fp32 V-cycle)",         "tab:green",  "--"),
    ("cpu32.np288", "CPU: 288 cores (fp32 V-cycle)",         "tab:green",  ":"),
]
LINE = re.compile(r"^Best timings for ndof =\s*(\d+)\s+mv\s+([0-9.e+-]+)\s+mv smooth\s+([0-9.e+-]+)\s+cg-mg\s+([0-9.e+-]+)")

data = {}
for label, *_ in RUNS:
    path = os.path.join(rdir, f"{label}.{jobid}.out")
    if not os.path.exists(path):
        print(f"missing: {path}")
        continue
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE.match(line)
            if m:
                n, mv, mvs, cg = int(m[1]), float(m[2]), float(m[3]), float(m[4])
                rows.append((n, mv, mvs, cg))
    data[label] = rows
    print(f"{label}: {len(rows)} sizes, largest {rows[-1][0] if rows else 0}")

with open(os.path.join(rdir, f"gpu_vs_cpu.{jobid}.csv"), "w") as f:
    f.write("run,ndofs,mv_s,mv_smooth_s,cg_mg_s,mv_dofs_per_s,smooth_dofs_per_s,cg_mg_dofs_per_s\n")
    for label, rows in data.items():
        for n, mv, mvs, cg in rows:
            f.write(f"{label},{n},{mv},{mvs},{cg},{n/mv:.4g},{n/mvs:.4g},{n/cg:.4g}\n")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True)
titles = ["matrix-vector product (fp64)", "smoother matrix-vector product", "CG with multigrid: full solve"]
for ax, idx, title in zip(axes, (1, 2, 3), titles):
    for label, legend, color, ls in RUNS:
        rows = data.get(label)
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[0] / r[idx] for r in rows]
        ax.loglog(xs, ys, marker="o", ms=3, color=color, ls=ls, lw=1.8, label=legend)
    ax.set_title(title)
    ax.set_xlabel("degrees of freedom")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("throughput [DoF/s]")
axes[2].legend(fontsize=7, loc="lower right", ncol=2)
fig.suptitle(f"Poisson, Q4, 3D, 5+5 Chebyshev smoothing, GH200 node of JUPITER, deal.II dealii-neon (job {jobid})")
fig.tight_layout()
out = os.path.join(rdir, f"gpu_vs_cpu.{jobid}.png")
fig.savefig(out, dpi=150)
print("wrote", out)
