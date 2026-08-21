from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/home/user/jlwang/models/lmcache/kvweave/lmcache-main/artifacts/cache_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


THEME = {
    "fig_bg": "#f7f6f2",
    "panel_bg": "#fcfbf8",
    "grid": "#d9d2c2",
    "text": "#222222",
    "muted": "#666666",
    "load": "#1f7a8c",
    "store": "#d1495b",
    "noquant": "#c8d5b9",
    "quant": "#2f3e46",
    "repeat_span": "#efe8d8",
}


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["fig_bg"],
            "axes.facecolor": THEME["panel_bg"],
            "axes.edgecolor": "#c9c2b2",
            "axes.labelcolor": THEME["text"],
            "xtick.color": THEME["text"],
            "ytick.color": THEME["text"],
            "text.color": THEME["text"],
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def scenario_data():
    two_q_no_quant = [
        {"req": "req1", "load": 0, "store": 40960, "ttft": 31142.89},
        {"req": "req2", "load": 0, "store": 40960, "ttft": 30843.99},
        {"req": "req1", "load": 0, "store": 40960, "ttft": 30574.52},
        {"req": "req2", "load": 0, "store": 40960, "ttft": 30684.99},
    ]
    two_q_quant = [
        {"req": "req1", "load": 0, "store": 40960, "ttft": 40969.26},
        {"req": "req2", "load": 0, "store": 40960, "ttft": 40448.05},
        {"req": "req1", "load": 40960, "store": 0, "ttft": 10993.73},
        {"req": "req2", "load": 40960, "store": 0, "ttft": 12324.27},
    ]
    three_q_no_quant = [
        {"req": "req1", "load": 0, "store": 40960, "ttft": 30655.75},
        {"req": "req2", "load": 0, "store": 40960, "ttft": 30645.98},
        {"req": "req3", "load": 0, "store": 40960, "ttft": 30619.58},
        {"req": "req1", "load": 0, "store": 40960, "ttft": 30776.84},
        {"req": "req2", "load": 0, "store": 40960, "ttft": 30746.89},
        {"req": "req3", "load": 0, "store": 40960, "ttft": 30704.48},
    ]
    three_q_quant = [
        {"req": "req1", "load": 0, "store": 40960, "ttft": 41779.65},
        {"req": "req2", "load": 0, "store": 40960, "ttft": 44026.81},
        {"req": "req3", "load": 0, "store": 40960, "ttft": 40856.67},
        {"req": "req1", "load": 9216, "store": 31744, "ttft": 35309.64},
        {"req": "req2", "load": 9216, "store": 31744, "ttft": 36021.95},
        {"req": "req3", "load": 9216, "store": 31744, "ttft": 36369.96},
    ]
    return {
        "2Q": {"NoQuant": two_q_no_quant, "Quant": two_q_quant},
        "3Q": {"NoQuant": three_q_no_quant, "Quant": three_q_quant},
    }


def retention(load: float, store: float) -> float:
    total = load + store
    return 0.0 if total == 0 else load / total


def plot_timeline(data):
    _apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle(
        "Cache Reuse Timeline: NoQuant vs Quant",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        "Second round behavior is highlighted to show whether cache is preserved.",
        ha="center",
        fontsize=10,
        color=THEME["muted"],
    )

    row_names = ["2Q", "3Q"]
    col_names = ["NoQuant", "Quant"]

    for r, row_name in enumerate(row_names):
        for c, col_name in enumerate(col_names):
            ax = axes[r, c]
            series = data[row_name][col_name]
            x = np.arange(len(series))
            labels = [f"{d['req']}-{i+1}" for i, d in enumerate(series)]
            loads = [d["load"] for d in series]
            stores = [d["store"] for d in series]

            split = len(series) // 2
            ax.axvspan(
                split - 0.5,
                len(series) - 0.5,
                color=THEME["repeat_span"],
                alpha=0.55,
                zorder=0,
            )

            ax.bar(
                x,
                loads,
                label="load tokens",
                color=THEME["load"],
                edgecolor="#ffffff",
                linewidth=1.0,
                zorder=3,
            )
            ax.bar(
                x,
                stores,
                bottom=loads,
                label="store tokens",
                color=THEME["store"],
                edgecolor="#ffffff",
                linewidth=1.0,
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(f"{row_name} repeated asks | {col_name}")
            ax.set_ylabel("Tokens")
            ax.grid(axis="y", color=THEME["grid"], alpha=0.5, linestyle="--", linewidth=0.8)
            ax.set_axisbelow(True)

            ax.axvline(split - 0.5, color="#7a6f58", linestyle="--", linewidth=1.2)
            y_top = max(np.array(loads) + np.array(stores))
            ax.text(
                split - 0.42,
                y_top * 1.025,
                "repeat round",
                fontsize=9,
                color="#5b5243",
            )

            repeat_rows = series[split:]
            repeat_rate = np.mean(
                [retention(d["load"], d["store"]) for d in repeat_rows]
            )
            status = "cache reused" if repeat_rate > 0 else "cache cleared"
            badge_color = "#1f7a8c" if repeat_rate > 0 else "#8d3b46"
            ax.text(
                0.03,
                0.88,
                f"{status}\nrepeat retention: {repeat_rate:.1%}",
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": badge_color,
                    "edgecolor": "none",
                    "alpha": 0.95,
                },
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.915))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = OUT_DIR / "cache_timeline_load_store.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_retention(data):
    _apply_theme()
    categories = ["2Q repeat", "3Q repeat"]
    no_quant_rates = []
    quant_rates = []

    two_no = data["2Q"]["NoQuant"][2:]
    two_q = data["2Q"]["Quant"][2:]
    three_no = data["3Q"]["NoQuant"][3:]
    three_q = data["3Q"]["Quant"][3:]

    no_quant_rates.append(np.mean([retention(d["load"], d["store"]) for d in two_no]))
    quant_rates.append(np.mean([retention(d["load"], d["store"]) for d in two_q]))

    no_quant_rates.append(np.mean([retention(d["load"], d["store"]) for d in three_no]))
    quant_rates.append(np.mean([retention(d["load"], d["store"]) for d in three_q]))

    x = np.arange(len(categories))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars1 = ax.bar(
        x - width / 2,
        no_quant_rates,
        width,
        label="NoQuant",
        color=THEME["noquant"],
        edgecolor="#ffffff",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x + width / 2,
        quant_rates,
        width,
        label="Quant",
        color=THEME["quant"],
        edgecolor="#ffffff",
        linewidth=1.2,
    )

    ax.set_title("Cache Retention Rate on Repeated Requests", fontsize=16, fontweight="bold")
    ax.set_ylabel("Retention rate (load / (load + store))")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=THEME["grid"], alpha=0.5, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.axhline(0.5, color="#9d9588", linewidth=1.0, linestyle=":")
    ax.text(
        1.42,
        0.515,
        "50% reference",
        fontsize=9,
        color=THEME["muted"],
    )

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    for i, (nq, q) in enumerate(zip(no_quant_rates, quant_rates, strict=True)):
        delta = q - nq
        ax.annotate(
            f"+{delta:.1%}",
            xy=(x[i] + width / 2, q),
            xytext=(x[i], min(1.03, q + 0.14)),
            arrowprops={
                "arrowstyle": "->",
                "color": THEME["quant"],
                "lw": 1.2,
            },
            ha="center",
            fontsize=10,
            color=THEME["quant"],
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.94,
        "Quant preserves cache across repeated asks\nwhile NoQuant tends to rebuild from scratch.",
        transform=ax.transAxes,
        fontsize=10,
        color=THEME["text"],
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f0ede6",
            "edgecolor": "#d2cab8",
        },
    )

    out = OUT_DIR / "cache_retention_rate_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_ttft_sequence(data):
    _apply_theme()
    fig, ax = plt.subplots(1, 1, figsize=(15, 8.6))
    fig.suptitle("Sequential TTFT Timeline in One Chart", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.945,
        "Same request order in one timeline: req1 -> req2 -> req1 -> req2 (each starts after previous finishes).",
        ha="center",
        fontsize=10,
        color=THEME["muted"],
    )

    panels = [
        ("NoQuant", data["2Q"]["NoQuant"], "#7f8c8d", 3.6),
        ("Quant", data["2Q"]["Quant"], THEME["quant"], 1.2),
    ]

    max_total = 0.0

    lane_positions: list[float] = []
    lane_labels: list[str] = []

    for name, series, color, base_y in panels:
        starts = []
        ends = []
        current = 0.0
        occurrence_counter = {"req1": 0, "req2": 0}
        occurrence_meta: dict[str, list[tuple[float, float]]] = {"req1": [], "req2": []}
        for d in series:
            starts.append(current)
            current += d["ttft"]
            ends.append(current)
        max_total = max(max_total, ends[-1])

        req_lane = {"req1": base_y + 0.55, "req2": base_y - 0.55}
        lane_positions.extend([base_y + 0.55, base_y - 0.55])
        lane_labels.extend([f"{name} req1", f"{name} req2"])

        for i, d in enumerate(series):
            duration = d["ttft"]
            y = req_lane[d["req"]]
            occurrence_counter[d["req"]] += 1
            occ = occurrence_counter[d["req"]]
            ax.broken_barh(
                [(starts[i], duration)],
                (y - 0.24, 0.48),
                facecolors=color,
                edgecolors="#ffffff",
                linewidth=1.5,
                alpha=0.95,
            )
            mid = starts[i] + duration / 2
            ax.text(
                mid,
                y,
                f"Run {occ}\n{duration/1000:.2f}s",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
            occurrence_meta[d["req"]].append((mid, y))
            if i > 0:
                ax.axvline(starts[i], color="#c0b8a8", linestyle="--", linewidth=1)

        for req_name, points in occurrence_meta.items():
            if len(points) >= 2:
                (x1, y1), (x2, y2) = points[0], points[1]
                ax.annotate(
                    "",
                    xy=(x2 - 450, y2 + 0.02),
                    xytext=(x1 + 450, y1 + 0.02),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "lw": 1.8,
                        "connectionstyle": "arc3,rad=0.0",
                    },
                )
                ax.text(
                    (x1 + x2) / 2,
                    (y1 + 0.32) if req_name == "req1" else (y1 - 0.36),
                    f"{req_name} repeat",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "#f7f4ec",
                        "edgecolor": "none",
                        "alpha": 0.9,
                    },
                )

        total = ends[-1]
        ax.text(
            max_total * 0.995,
            base_y,
            f"{name} total: {total/1000:.2f}s",
            ha="right",
            va="center",
            fontsize=10,
            color=THEME["text"],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#f0ede6",
                "edgecolor": "#d2cab8",
            },
        )

        repeat_req1 = series[2]["ttft"]
        repeat_req2 = series[3]["ttft"]
        ax.text(
            max_total * 1.005,
            base_y - 0.12,
            f"repeat req1 TTFT: {repeat_req1/1000:.2f}s\nrepeat req2 TTFT: {repeat_req2/1000:.2f}s",
            ha="left",
            va="center",
            fontsize=8.5,
            color=THEME["text"],
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": "#f0ede6",
                "edgecolor": "#d2cab8",
            },
        )

    ax.set_xlim(0, max_total * 1.2)
    ax.set_ylim(0.1, 4.4)
    ax.set_yticks(lane_positions)
    ax.set_yticklabels(lane_labels, fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", color=THEME["grid"], alpha=0.45, linestyle=":")
    ax.set_xlabel("Elapsed time (ms)")
    ax.set_axisbelow(True)
    ax.set_title("Two modes merged in one timeline (different req on separate lines)", loc="left", fontsize=12)

    ax.axhline(2.4, color="#b9b09f", linewidth=1.0, linestyle="--")
    ax.margins(y=0.12)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT_DIR / "cache_ttft_sequence_2q.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    data = scenario_data()
    p1 = plot_timeline(data)
    p2 = plot_retention(data)
    p3 = plot_ttft_sequence(data)
    print(f"generated: {p1}")
    print(f"generated: {p2}")
    print(f"generated: {p3}")


if __name__ == "__main__":
    main()
