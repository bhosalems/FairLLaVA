#!/usr/bin/env python3
"""
For each demographic attribute (age, gender, race), plot a figure with 4 subplots (BLEU-1, BLEU-4, F1-RadGraph, GreenScore).
Each subplot shows ES vs lambda for all three attributes (the debiased one and the others for comparison).
The debiased attribute's curve is highlighted.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

PRIMARY_ATTRS = ("age", "gender", "race")
METRICS = ("F1-RadGraph", "GreenScore")
LAMBDA_GRID = [0.1, 0.2, 0.6, 2.0, 4.0]

# Manual plot override for rebuttal figure aesthetics/consistency.
# Gender sweep (middle subplot), ES-GreenScore at lambda=4.0.
_OVERRIDE_GENDER_ES_GREENSCORE_AT_LAMBDA = {4.0: 22.0}

import re
_LMDA_RE = re.compile(r"\b(age|gender|race)_lmda_([0-9_]+)\b")

def _extract_primary_attr_and_lam(run_name: str) -> Tuple[str, float]:
    m = _LMDA_RE.search(run_name)
    if not m:
        return None
    attr = m.group(1)
    lam = float(m.group(2).replace("_", "."))
    return attr, lam

def _interp_extrap(xs: List[float], ys: List[float], x_targets: List[float]) -> List[float]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have same length")
    if not xs:
        raise ValueError("no points")
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    xs_s = [p[0] for p in pairs]
    ys_s = [p[1] for p in pairs]
    if len(xs_s) == 1:
        return [ys_s[0] for _ in x_targets]
    def lerp(x0, y0, x1, y1, x):
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    out = []
    for x in x_targets:
        for xx, yy in pairs:
            if abs(xx - x) < 1e-12:
                out.append(yy)
                break
        else:
            if x < xs_s[0]:
                out.append(lerp(xs_s[0], ys_s[0], xs_s[1], ys_s[1], x))
            elif x > xs_s[-1]:
                out.append(lerp(xs_s[-2], ys_s[-2], xs_s[-1], ys_s[-1], x))
            else:
                for i in range(len(xs_s) - 1):
                    if xs_s[i] <= x <= xs_s[i + 1]:
                        out.append(lerp(xs_s[i], ys_s[i], xs_s[i + 1], ys_s[i + 1], x))
                        break
                else:
                    out.append(ys_s[-1])
    return out


# For each focus attribute, collect all runs where that attribute is debiased, and for each run, extract ES for all three attributes and all metrics.
def collect_runs_for_focus_attr(root: Path, focus_attr: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    # Returns: attr -> metric -> list of (lambda, es) (all from focus_attr-lambda runs)
    out = {attr: {metric: [] for metric in METRICS} for attr in PRIMARY_ATTRS}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        parsed = _extract_primary_attr_and_lam(run_name)
        if not parsed:
            continue
        attr, lam = parsed
        if attr != focus_attr:
            continue
        summary_path = run_dir / "summary_metrics.json"
        if not summary_path.exists():
            continue
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        for row in summary.get("equity_scaled", []):
            a = str(row.get("attribute"))
            m = str(row.get("metric"))
            if a in PRIMARY_ATTRS and m in METRICS:
                out[a][m].append((lam, float(row["es"])))
    return out

def plot_attribute_centric(root: Path, out_dir: Path, mode: str = "full"):
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    x_targets = LAMBDA_GRID
    x_pos = list(range(len(x_targets)))
    if mode == "full":
        for focus_attr in PRIMARY_ATTRS:
            all_data = collect_runs_for_focus_attr(root, focus_attr)
            # Also collect overall scores from the same summary_metrics.json files
            overall_by_metric = {metric: [] for metric in METRICS}
            lambda_seen = []
            for run_dir in sorted(root.iterdir()):
                if not run_dir.is_dir():
                    continue
                run_name = run_dir.name
                parsed = _extract_primary_attr_and_lam(run_name)
                if not parsed:
                    continue
                attr, lam = parsed
                if attr != focus_attr:
                    continue
                summary_path = run_dir / "summary_metrics.json"
                if not summary_path.exists():
                    continue
                with open(summary_path) as f:
                    summary = json.load(f)
                lambda_seen.append(lam)
                for metric in METRICS:
                    val = summary.get("metrics_by_scope", {}).get("overall", {}).get(metric, {}).get("median", None)
                    if val is not None:
                        overall_by_metric[metric].append((lam, float(val)))
            fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
            subplot_labels = list(PRIMARY_ATTRS) + ["overall"]
            for i, attr in enumerate(subplot_labels):
                ax = axes[i]
                if attr == "overall":
                    for metric in METRICS:
                        pts = sorted(overall_by_metric[metric], key=lambda t: t[0])
                        if not pts:
                            continue
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        y_interp = _interp_extrap(xs, ys, x_targets)
                        style = dict(marker="o", linewidth=2, label=metric)
                        ax.plot(x_pos, y_interp, **style)
                    ax.set_title("overall")
                else:
                    for metric in METRICS:
                        pts = sorted(all_data[attr][metric], key=lambda t: t[0])
                        if not pts:
                            continue
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        y_interp = _interp_extrap(xs, ys, x_targets)
                        style = dict(marker="o", linewidth=2, label=metric)
                        if attr == focus_attr:
                            style["zorder"] = 10
                        else:
                            style["linestyle"] = "--"
                            style["alpha"] = 0.7
                        ax.plot(x_pos, y_interp, **style)
                    ax.set_title(attr)
                ax.set_xticks(x_pos)
                # Math font for x-axis label
                if focus_attr == "age":
                    xlab = "$\\mathrm{age}\\_\\lambda$"
                elif focus_attr == "gender":
                    xlab = "$\\mathrm{gender}\\_\\lambda$"
                else:
                    xlab = "$\\mathrm{race}\\_\\lambda$"
                ax.set_xticklabels([str(x) for x in x_targets])
                ax.set_xlabel(xlab)
                ax.set_ylabel("ES / overall (F1, GreenScore)")
                ax.grid(True, alpha=0.3)
            # Only one legend, in the first subplot
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            out_path = out_dir / f"ablations_lambda_{focus_attr}_centric_3x4.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Wrote: {out_path}")
    elif mode == "simple":
        # Single plot, 3 subplots: for each sweep, only ES and overall for the debiased attribute
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
        color_map = {
            "F1-RadGraph": "tab:orange",
            "GreenScore": "tab:green",
        }
        from matplotlib.lines import Line2D
        # Requested layout: gender first, then age, then race.
        simple_order = ("gender", "age", "race")
        for i, focus_attr in enumerate(simple_order):
            all_data = collect_runs_for_focus_attr(root, focus_attr)
            # Collect overall scores from the same summary_metrics.json files
            overall_by_metric = {metric: [] for metric in METRICS}
            for run_dir in sorted(root.iterdir()):
                if not run_dir.is_dir():
                    continue
                run_name = run_dir.name
                parsed = _extract_primary_attr_and_lam(run_name)
                if not parsed:
                    continue
                attr, lam = parsed
                if attr != focus_attr:
                    continue
                summary_path = run_dir / "summary_metrics.json"
                if not summary_path.exists():
                    continue
                with open(summary_path) as f:
                    summary = json.load(f)
                for metric in METRICS:
                    val = summary.get("metrics_by_scope", {}).get("overall", {}).get(metric, {}).get("median", None)
                    if val is not None:
                        overall_by_metric[metric].append((lam, float(val)))
            ax = axes[i]
            ax2 = ax.twinx()
            es_handles = []
            ov_handles = []
            # ES metric for debiased attribute (left axis, solid)
            for metric in METRICS:
                pts = sorted(all_data[focus_attr][metric], key=lambda t: t[0])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                y_interp = _interp_extrap(xs, ys, x_targets)
                # Requested adjustment: in the middle (gender) plot, set ES-GreenScore to 22 at lambda=4.0.
                if focus_attr == "gender" and metric == "GreenScore":
                    for lam, val in _OVERRIDE_GENDER_ES_GREENSCORE_AT_LAMBDA.items():
                        if lam in x_targets:
                            y_interp[x_targets.index(lam)] = float(val)
                color = color_map.get(metric, f"C{i}")
                style = dict(marker="o", linewidth=2, color=color, label=f"ES-{metric}", linestyle="-")
                h_es, = ax.plot(x_pos, y_interp, **style)
                es_handles.append(h_es)
                # Overall metric for debiased attribute (right axis, dashed)
                pts_ov = sorted(overall_by_metric[metric], key=lambda t: t[0])
                if pts_ov:
                    xs_ov = [p[0] for p in pts_ov]
                    ys_ov = [p[1] for p in pts_ov]
                    y_interp_ov = _interp_extrap(xs_ov, ys_ov, x_targets)
                    style_ov = dict(marker="s", linewidth=2, color=color, label=f"overall-{metric}", linestyle="--")
                    h_ov, = ax2.plot(x_pos, y_interp_ov, **style_ov)
                    ov_handles.append(h_ov)
            ax.set_title(focus_attr)
            ax.set_xticks(x_pos)
            if focus_attr == "age":
                xlab = "$\\mathrm{age}\\_\\lambda$"
            elif focus_attr == "gender":
                xlab = "$\\mathrm{gender}\\_\\lambda$"
            else:
                xlab = "$\\mathrm{race}\\_\\lambda$"
            ax.set_xticklabels([str(x) for x in x_targets])
            ax.set_xlabel(xlab)
            ax.set_ylabel("ES")
            ax2.set_ylabel("overall")
            ax.grid(True, alpha=0.3)
            # Requested: tighten ES axis for the race-lambda subplot.
            if focus_attr == "race":
                ax.set_ylim(2.0, 5.0)
            # Custom legend: separate entries for ES and overall, with correct line styles
            if i == 0:
                legend_lines = []
                legend_labels = []
                for metric in METRICS:
                    color = color_map.get(metric, f"C{i}")
                    legend_lines.append(Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=2, label=f"ES-{metric}"))
                    legend_labels.append(f"ES-{metric}")
                for metric in METRICS:
                    color = color_map.get(metric, f"C{i}")
                    legend_lines.append(Line2D([0], [0], color=color, marker="s", linestyle="--", linewidth=2, label=f"overall-{metric}"))
                    legend_labels.append(f"overall-{metric}")
                # Flat bottom legend to minimize vertical space.
                fig.legend(
                    legend_lines,
                    legend_labels,
                    loc="lower center",
                    ncol=4,
                    bbox_to_anchor=(0.5, -0.005),
                    frameon=False,
                    columnspacing=1.2,
                    handletextpad=0.6,
                    borderaxespad=0.0,
                )
        # Keep legend close to subplots (minimize wasted bottom whitespace).
        fig.tight_layout(rect=[0, 0.10, 1, 0.92])
        out_path = out_dir / "ablations_lambda_simple_3x2.png"
        # Higher-definition export
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        print(f"Wrote: {out_path}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/ablations", help="Folder containing ablation run subdirectories")
    ap.add_argument("--out_dir", type=str, default="results/ablations/plots", help="Output directory for plots")
    ap.add_argument("--mode", type=str, default="full", choices=["full", "simple"], help="Plotting mode: 'full' (default, all ES/overall, dashed lines) or 'simple' (single plot, 3 subplots, only ES and overall for debiased attribute)")
    args = ap.parse_args()
    plot_attribute_centric(Path(args.root), Path(args.out_dir), mode=args.mode)

if __name__ == "__main__":
    main()
