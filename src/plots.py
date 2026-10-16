"""Simple charts for the notebooks (matplotlib), in one consistent style."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical palette in a fixed order (the first 3 colours are also readable with colour blindness)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax, grid_axis):
    """Shared, muted look of the axes: no top/right frame, light grid lines.

    grid_axis="y" draws horizontal grid lines (for line charts),
    grid_axis="x" draws vertical ones (for horizontal bar charts).
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def share_by_year(df, value_col, value, group_col="Source", title=""):
    """Lines: share of rows with value_col == value in each year, one line per group."""
    table = (df[value_col] == value).groupby([df["Year"], df[group_col]]).mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4))
    for color, group in zip(SERIES_COLORS, table.columns):
        ax.plot(table.index, table[group] * 100, color=color, linewidth=2, marker="o",
                markersize=5, label=group)
        last_year = table[group].last_valid_index()
        ax.annotate(group, (last_year, table.loc[last_year, group] * 100), xytext=(6, 0),
                    textcoords="offset points", color=TEXT_MUTED, va="center", fontsize=9)
    ax.set_ylabel("% of rows", color=TEXT_MUTED)
    ax.set_title(title, color=TEXT, loc="left")
    ax.legend(frameon=False)
    _style(ax, grid_axis="y")
    plt.tight_layout()
    return fig


def metric_with_error(summary, metric="macro_f1", title=""):
    """Horizontal bars: mean of a metric from CV ± standard deviation, sorted."""
    data = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1))
    ax.barh(data["model"], data[metric], xerr=data[metric + "_std"], color=SERIES_COLORS[0],
            height=0.6, error_kw={"ecolor": TEXT_MUTED, "capsize": 3})
    for y, value in enumerate(data[metric]):
        ax.text(value + 0.01, y, f"{value:.3f}", va="center", color=TEXT_MUTED, fontsize=9)
    ax.set_xlabel(f"{metric} (CV mean ± SD)", color=TEXT_MUTED)
    ax.set_title(title, color=TEXT, loc="left")
    _style(ax, grid_axis="x")
    plt.tight_layout()
    return fig


def confusion_heatmap(confusion, title=""):
    """Confusion matrix normalised by row (= recall on the diagonal), with counts."""
    shares = confusion.div(confusion.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(shares.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            color = "white" if shares.iloc[i, j] > 0.6 else TEXT
            ax.text(j, i, f"{shares.iloc[i, j]:.0%}\n({confusion.iloc[i, j]:,})",
                    ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(len(shares.columns)), shares.columns)
    ax.set_yticks(range(len(shares.index)), shares.index)
    ax.set_xlabel("predicted", color=TEXT_MUTED)
    ax.set_ylabel("actual", color=TEXT_MUTED)
    ax.set_title(title, color=TEXT, loc="left")
    plt.tight_layout()
    return fig


def importance_by_class(table, top=12, title=""):
    """Horizontal bars of mean |SHAP| for the most important features, separately for each class."""
    data = table.head(top).iloc[::-1]
    classes = [c for c in data.columns if c != "mean_all_classes"]
    y = np.arange(len(data))
    height = 0.8 / len(classes)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1))
    for k, (color, cls) in enumerate(zip(SERIES_COLORS, classes)):
        # first class at the top of each group, in the same order as in the legend
        ax.barh(y - k * height, data[cls], height=height * 0.9, color=color, label=cls)
    ax.set_yticks(y - height * (len(classes) - 1) / 2, data.index)
    ax.set_xlabel("mean |SHAP| (log-odds)", color=TEXT_MUTED)
    ax.set_title(title, color=TEXT, loc="left")
    ax.legend(frameon=False, loc="lower right")
    _style(ax, grid_axis="x")
    plt.tight_layout()
    return fig
