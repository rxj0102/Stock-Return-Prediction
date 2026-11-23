"""Run the walk-forward study on the frozen data and write results/.

  results/metrics.json            everything the README reports
  results/forecasts_<series>.csv.gz  every forecast next to the return it was for
  results/figures/rolling_r2.png  ten year out-of-sample R2 through time
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import study  # noqa: E402
from src.data import SERIES, load  # noqa: E402

LABEL = {"market": "Market", "large": "Largest tenth", "small": "Smallest tenth"}


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def block(f, cfg) -> dict:
    y = f["target"].to_numpy()
    out = {"n_days": int(len(f)), "buy_hold_sharpe": study.sharpe(y), "up_days": float((y > 0).mean()),
           "autocorr_1": float(np.corrcoef(y[1:], y[:-1])[0, 1]), "models": {}}
    for m in study.MODELS[1:]:
        p = f[m].to_numpy()
        out["models"][m] = {**study.forecast_stats(p, y, cfg), **study.trading_stats(p, y, cfg)}
        if m in ("Ridge", "Boosting"):
            out["models"][m]["vs_ar1_t"] = study.loss_diff_t(p, f["AR(1)"].to_numpy(), y, cfg)
    return out


def figure(forecasts: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    colors = {"AR(1)": "#8a8980", "Ridge": "#eb6834", "Boosting": "#2a78d6"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, s in zip(axes, SERIES):
        f = forecasts[s]
        for m, c in colors.items():
            r2 = study.rolling_r2(f[m], f["target"]) * 100
            ax.plot(r2.index, r2, color=c, linewidth=1.6, label=m)
        ax.axhline(0, color=ink, linewidth=0.8)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_title(LABEL[s], loc="left", fontsize=10.5)
    axes[0].set_ylabel("R2, percent", color=muted)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Out-of-sample R2 over the trailing ten years, against always predicting zero",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "rolling_r2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = study.Config()
    data = load()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    forecasts, series = {}, {}
    for s in SERIES:
        f = study.walk_forward(data[s], cfg)
        forecasts[s] = f
        f.round(8).to_csv(out / f"forecasts_{s}.csv.gz", index_label="date",
                          compression={"method": "gzip", "mtime": 0})
        r2 = study.rolling_r2(f["AR(1)"], f["target"])
        below = r2.index[(r2 < 0).to_numpy()]
        series[s] = {"label": LABEL[s], "full": block(f, cfg),
                     "ar1_trailing_r2_first_below_zero": str(below[0].date()) if len(below) else None,
                     "ar1_trailing_r2_peak": float(r2.max()), "ar1_trailing_r2_peak_date": str(r2.idxmax().date()),
                     "eras": {lab: block(f.loc[a:b], cfg) for lab, a, b in study.ERAS},
                     "tick_regimes": {lab: block(f.loc[a:b], cfg) for lab, a, b in study.TICK_REGIMES}}
        print(s, "done", flush=True)
    first = forecasts["market"]
    payload = rounded({
        "sample": {"data_first": str(data.index[0].date()), "first_forecast": str(first.index[0].date()),
                   "last_forecast": str(first.index[-1].date()), "n_forecasts": int(len(first)),
                   "years": float(len(first) / 252), "train_days": cfg.train, "refit_days": cfg.refit,
                   "n_refits": int(np.ceil(len(first) / cfg.refit)), "n_features": len(study.FEATURES),
                   "cost_bps": cfg.cost * 1e4},
        "series": series,
    })
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    figure(forecasts, out / "figures")
    for s in SERIES:
        for era, b in [("full", series[s]["full"])] + list(series[s]["eras"].items()):
            print(f"{s:7s}{era:10s} n={b['n_days']:5d} BH {b['buy_hold_sharpe']:.2f} ac {b['autocorr_1']:+.3f}")
            for m, v in b["models"].items():
                d = v["net_minus_buy_hold"]
                print(f"    {m:9s} R2 {v['r2_oos']*100:+6.2f}% CW {v['clark_west_t']:+6.2f} dir {v['direction']:.3f} "
                      f"SRg {v['sharpe_gross']:+.2f} SRn {v['sharpe_net']:+.2f} be {v['break_even_cost_bps']:5.1f} "
                      f"diff {d['diff']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] vsAR {v.get('vs_ar1_t', float('nan')):+.2f}")


if __name__ == "__main__":
    main()
