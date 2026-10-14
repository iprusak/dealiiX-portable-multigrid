#!/usr/bin/env python3
"""Turn hybrid / bk3 output files into one CSV.

  parse_hybrid.py out1.out [out2.out ...] > summary.csv

Hybrid runs are split at the "=== k=<k> ... ./program k degree n_refine
switch_level n_subdivisions [n_smooth [stretch]]" header lines written by the
run_sbatch scripts. poisson_cube_bk3 / poisson_cube_cpu outputs (the
"n_proc cells dofs mv_outer mv_inner cg_time cg_its cg_reduction" table) give
rows with kind = bk3 and k = 0, switch_level = -1.

CG times: min and median over the solves after the first one (the first is a
warm-up). Level times are the printed sums over the V-cycles of the last solve.
"""

import csv
import re
import statistics
import sys

NUM = r"[\d.]+(?:e[+-]?\d+)?"

HEADER = re.compile(
    r"^=+ k=(\d+):? (?:\S+ )?\./program\S* (\d+) (\d+) (\d+) (\d+) (\d+)(?: (\d+))?(?: (-?\d+))?",
    re.M,
)
CG = re.compile(
    r"CG on the fine level \((\d+) DoFs\): (\d+) iterations, reduction (" + NUM + r") per iteration, (" + NUM + r") s"
)
SWITCH = re.compile(
    r"==== switch, rank 0: (\d+) calls, bcast (" + NUM + r") device->host (" + NUM + r") CPU part (" + NUM +
    r") host->device (" + NUM + r")"
)
PREC = re.compile(r"Coarse solver (\d+) times: (" + NUM + r") tot prec (" + NUM + r")")
LEVEL = re.compile(r"^L(\d+)\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s*$", re.M)
BK3_ROW = re.compile(
    r"^(\d+)\s+(\d+)\s+(\d+)\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s+(\d+)\s+(" + NUM + r")\s*$", re.M
)

FIELDS = [
    "file", "kind", "k", "degree", "n_refine", "stretch", "cycle", "switch_level", "n_smooth", "dofs", "its",
    "reduction", "cg_min", "cg_median", "gpu_prec", "cpu_prec", "cpu_part", "d2h", "h2d", "bcast",
    "gpu_levels", "cpu_levels",
]


def levels(block):
    """'L3:smoother/mv/restrict/prolongate;...' of one 'levels' table."""
    return ";".join(
        "L%s:%s/%s/%s/%s" % m.groups() for m in LEVEL.finditer(block)
    )


def hybrid_rows(name, text):
    heads = list(HEADER.finditer(text))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        run = text[h.start():end]
        k, _, degree, n_refine, switch, _, n_smooth, stretch = h.groups()
        k, degree, n_refine, switch = int(k), int(degree), int(n_refine), int(switch)
        n_smooth = int(n_smooth) if n_smooth else 3
        stretch = int(stretch) if stretch else -1
        cg = CG.findall(run)
        if not cg:
            print("no CG result: %s k=%d s=%d" % (name, k, switch), file=sys.stderr)
            continue
        times = [float(c[3]) for c in cg][1:] or [float(cg[0][3])]
        gpu_blk = cpu_blk = ""
        a, b, c = (run.find("==== GPU side levels"), run.find("==== CPU side levels"),
                   run.find("==== switch"))
        if a >= 0 and b > a:
            gpu_blk = run[a:b]
        if b >= 0 and c > b:
            cpu_blk = run[b:c]
        sw = SWITCH.search(run)
        gp = PREC.search(gpu_blk)
        cp = PREC.search(cpu_blk)
        yield {
            "file": name, "kind": "hybrid", "k": k, "degree": degree, "n_refine": n_refine,
            "stretch": stretch, "cycle": 3 * n_refine + stretch if stretch >= 0 else "",
            "switch_level": switch, "n_smooth": n_smooth, "dofs": cg[0][0], "its": cg[0][1],
            "reduction": cg[0][2], "cg_min": min(times), "cg_median": statistics.median(times),
            "gpu_prec": gp.group(3) if gp else "", "cpu_prec": cp.group(3) if cp else "",
            "cpu_part": sw.group(4) if sw else "", "d2h": sw.group(3) if sw else "",
            "h2d": sw.group(5) if sw else "", "bcast": sw.group(2) if sw else "",
            "gpu_levels": levels(gpu_blk), "cpu_levels": levels(cpu_blk),
        }


def bk3_rows(name, text):
    seen = set()
    for m in BK3_ROW.finditer(text):
        n_proc, cells, dofs, _, _, cg_time, its, red = m.groups()
        if dofs in seen:  # the table is reprinted after every cycle
            continue
        seen.add(dofs)
        yield {
            "file": name, "kind": "bk3", "k": 0, "dofs": dofs, "its": its, "reduction": red,
            "cg_min": cg_time, "cg_median": cg_time, "switch_level": -1,
        }


def main(paths):
    out = csv.DictWriter(sys.stdout, fieldnames=FIELDS, restval="")
    out.writeheader()
    for path in paths:
        text = open(path, errors="replace").read()
        for row in hybrid_rows(path, text):
            out.writerow(row)
        if not HEADER.search(text):
            for row in bk3_rows(path, text):
                out.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
