import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

mirage_constants = {
    "MedGemma 27B": {
        "P0": 59.9,
        "P1": 61.7,
    },
    "Gemma-3 27B": {
        "P0": 55.3,
        "P1": 58.4,
    },
    "MedGemma 4B": {
        "P0": 47.2,
        "P1": 46.6,

    },
    "Gemma-3 4B": {
        "P0": 44.1,
        "P1": 40.2,
    },
    "Gemma-3 1B": {
        "P0": 34.6,
        "P1": 27.7,
    },
}

pubmedqa_no_context_constants = {
    "MedGemma 27B": {
        "P0": 43.3,
    },
    "Gemma-3 27B": {
        "P0": 21.2,
    },
    "MedGemma 4B": {
        "P0": 33.4,
    },
    "Gemma-3 4B": {
        "P0": 24.2,
    },
    "Gemma-3 1B": {
        "P0": 40.2,
    },
}

pubmedqa_with_context_constants = {
    "MedGemma 27B": {
        "P0": 74.7,
    },
    "Gemma-3 27B": {
        "P0": 60.0,
    },
    "MedGemma 4B": {
        "P0": 73.1,
    },
    "Gemma-3 4B": {
        "P0": 60.3,
    },
    "Gemma-3 1B": {
        "P0": 58.5,
    },
}

medical_exam_constants = {
    "MedGemma 27B": {
        "P0": 48.1,
    },
    "Gemma-3 27B": {
        "P0": 48.1,
    },
    "MedGemma 4B": {
        "P0": 37.3,
    },
    "Gemma-3 4B": {
        "P0": 35.7,
    },
    "Gemma-3 1B": {
        "P0": 28.4,
    },
}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    return df


def make_line_plots(df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    datasets = sorted(df["dataset"].unique())
    n_panels = min(4, len(datasets))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    # Collect handles/labels for a shared legend
    legend_handles = []
    legend_labels = []

    ci_by_dataset = {
        "MIRAGE": 1.2,
        "PubMedQA (no context)": 2.9,
        "PubMedQA (with context)": 2.9,
        "Medical exam": 4.1,
    }

    # Calculate global y-axis limits across all datasets
    y_min = df["accuracy"].min()
    y_max = df["accuracy"].max()
    max_ci = max(ci_by_dataset.values())
    y_min = y_min - max_ci
    y_max = y_max + max_ci
    y_range = y_max - y_min
    y_min = y_min - 0.05 * y_range
    y_max = y_max + 0.05 * y_range

    for idx in range(4):
        ax = axes[idx]
        if idx >= n_panels:
            ax.axis("off")
            continue

        dataset_name = datasets[idx]
        subset = df[df["dataset"] == dataset_name].copy()
        
        # Add subplot label (A, B, C, D)
        subplot_label = chr(65 + idx)  # 65 is ASCII for 'A'
        ax.text(-0.1, 1.05, subplot_label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
        
        ax.set_title(dataset_name, fontsize=16, fontweight="bold")

        ci_width = ci_by_dataset.get(dataset_name)

        for (size, ft), group in subset.groupby(["size", "FT"]):
            group_sorted = group.sort_values("budget")
            if ft == 0:
                prefix = "Gemma-3"
            else:
                prefix = "MedGemma"
            size_str = f"{size}B"
            label = f"{prefix} {size_str}"
            if ci_width is not None:
                handle = ax.errorbar(
                    group_sorted["budget"],
                    group_sorted["accuracy"],
                    yerr=ci_width,
                    marker="o",
                    label=label,
                    linestyle="-",
                    capsize=3,
                )
            else:
                handle = ax.plot(
                    group_sorted["budget"],
                    group_sorted["accuracy"],
                    marker="o",
                    label=label,
                )[0]

            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(handle)

            # Add P0 horizontal lines for each dataset
            constants_map = {
                "MIRAGE": mirage_constants,
                "PubMedQA (no context)": pubmedqa_no_context_constants,
                "PubMedQA (with context)": pubmedqa_with_context_constants,
                "Medical exam": medical_exam_constants,
            }
            
            if dataset_name in constants_map and label in constants_map[dataset_name]:
                p0_value = constants_map[dataset_name][label]["P0"]
                color = handle.get_color() if hasattr(handle, 'get_color') else handle.lines[0].get_color()
                ax.axhline(y=p0_value, color=color, linestyle='--', linewidth=1, alpha=0.7)

        if idx in [2, 3]:
            ax.set_xlabel("Token budget", fontsize=14)
        if idx in [0, 2]:
            ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.set_xticks([128, 256, 512, 768, 1024])
        ax.set_xticklabels([128, 256, 512, 768, 1024])
        if idx >= 0:
            ax.set_ylim(20, 82)
        ax.tick_params(axis="both", which="major", labelsize=12)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Shared legend outside the axes, centered below the subplots
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(legend_labels),
            fontsize=12,
        )

    # fig.suptitle("Accuracy vs Budget by Dataset", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 2x2 line plots of accuracy vs budget by dataset."
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path(__file__).with_name("econ-of-acc-table.csv"),
        help="Path to econ-of-acc-table.csv",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(__file__).with_name("econ-of-acc-2x2.png"),
        help="Where to save the 2x2 panel figure (PNG).",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="If set, do not save to file; display the plot interactively instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.csv_path)
    if args.no_save:
        make_line_plots(df, output_path=None)
    else:
        make_line_plots(df, output_path=args.output_path)


if __name__ == "__main__":
    main()
