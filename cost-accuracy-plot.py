import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the CSV (adjust path if you run this from another directory)
df = pd.read_csv("econ-of-accuracy-full-table.csv")

# Make columns easier to work with
df = df.rename(columns={
    "tokens gen": "tokens_gen",  # optional, for clarity
})

# Create explicit micro-dollar column (1 dollar = 1e6 micro-dollars)
df["cost-micro-dollars"] = df["cost"] * 1_000_000

# Scale marker sizes so differences are visible
size_scale = 10
marker_sizes = df["size"] * size_scale

plt.figure(figsize=(8, 6))
ax = plt.gca()

scatter = ax.scatter(
    df["cost-micro-dollars"],  # x-axis
    df["accuracy"],            # y-axis
    s=marker_sizes,            # marker size from `size`
    c=df["Context"],           # color from binary Context
    cmap="bwr",                # blue/red for 0/1
    alpha=0.7,
    edgecolors="k",
)

# Legend for marker sizes (using min/median/max of `size`)
size_values = [
    int(df["size"].min()),
    int(df["size"].median()),
    int(df["size"].max()),
]
size_values = sorted(set(size_values))
size_handles = [
    ax.scatter(
        [],
        [],
        s=val * size_scale,
        c="none",
        edgecolors="k",
        label=f"size = {val}B",
    )
    for val in size_values
]

# Legend for Context color mapping
cmap = plt.cm.get_cmap("bwr")
color_handles = [
    mpatches.Patch(color=cmap(0.0), label="No context"),
    mpatches.Patch(color=cmap(1.0), label="With context"),
]

legend1 = ax.legend(
    handles=size_handles,
    title="Model size",
    loc="lower left",
    fontsize=10,
    title_fontsize=11,
)
ax.add_artist(legend1)

ax.legend(
    handles=color_handles,
    title="Context",
    loc="lower right",
    fontsize=10,
    title_fontsize=11,
)

ax.set_xlabel("Cost (micro-dollars) per question", fontsize=14)
ax.set_ylabel("Accuracy", fontsize=14)
ax.set_title(
    "Accuracy vs Estimated Cost",
    fontsize=16,
    fontweight="bold",
)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.tick_params(axis="both", which="major", labelsize=12)
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontweight("bold")
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontweight("bold")

# # Optional: colorbar to show mapping of color to Context value
# cbar = plt.colorbar(scatter)
# cbar.set_label("Context", fontsize=12)
# cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("figures/cost_accuracy_plot.pdf", dpi=300, bbox_inches='tight')
plt.close()