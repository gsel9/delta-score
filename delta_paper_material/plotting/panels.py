import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from sklearn.metrics import auc
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from . import plot_utils 
from .delta_plots import delta_score

plot_utils.setup()


def remove_ticks(axis):

    axis.set_yticks([])
    axis.set_xticks([])

    axis.set_yticklabels([])
    axis.set_xticklabels([])


def effect_noise_panel(datasets, filename):

    cls_sep = ["100%", "75%", "50%"]
    score_labels = ["True negative rate", "True positive rate", r"F$_2$ score"]
    
    fig, axes = plt.subplots(3, 3, figsize=plot_utils.set_fig_size(430, fraction=1.8, subplots=(3, 2.5)))
    fig.subplots_adjust(wspace=0, hspace=0) 
    fig.patch.set_linewidth(3)
    
    for row_num, score in enumerate(["tnr", "tpr", "fbeta_score"]):
        for col_num, overlap in enumerate([-2, -1, 0]):

            outer_axis = axes[row_num, col_num]
            outer_axis.patch.set_linewidth(4)

            remove_outer_spines(outer_axis, row_num, col_num)
            remove_ticks(outer_axis)

            inner_axis = inset_axes(outer_axis, width=2, height=1.2, loc="center")

            if col_num == 0:
                outer_axis.set_ylabel(score_labels[row_num], fontdict={"fontsize": 12})

            if row_num == 0:
                outer_axis.set_title(f"Class separability {cls_sep[col_num]}", pad=10)

            for i, (key, data) in enumerate(datasets.items()):
                _plot_groupwise(data.where(data.loc[:, "overlap"] == overlap).dropna(how="all"), "noise", key, inner_axis, score, i)

            plot_utils.format_axis(inner_axis, fig, ylim=[-0.05, 1.05], grid=True, xlabel="Label noise")
    
    fig.savefig(f"{filename}/panel_effect_noise.pdf", edgecolor="k", transparent=True, bbox_inches="tight") 


def _plot_groupwise(data, grouper, key, axis, score, i):

    labels = {
        "unweighted": "Uniform",
        "focal": "Focal",
        "delta": "Delta",
        "lmdnn": "Lmdnn",
        "focal_adaptive": "Focal-S",
        "delta_adaptive": "Delta-S",
        "lmdnn_adaptive": "Lmdnn-S"
    }

    groups = data.groupby(grouper)[score]

    scores_avg = groups.mean()
    scores_std = groups.std()

    axis.plot(scores_avg.index.values, scores_avg.values, marker="o", linestyle="", c=f"C{i}", label=labels[key], markersize=4, alpha=0.8)
    axis.plot(scores_avg.index.values, scores_avg.values, marker="", linestyle="-", c=f"C{i}", alpha=0.5)
    
    axis.fill_between(scores_avg.index.values, scores_avg.values - scores_std, scores_avg.values + scores_std, color=f"C{i}", alpha=0.1)
    axis.fill_between(scores_avg.index.values, scores_avg.values - scores_std, scores_avg.values + scores_std, color=f"C{i}", alpha=0.1)

    axis.set_xticks(scores_avg.index.values)
    axis.set_xticklabels(scores_avg.index.values)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, columnspacing=0.1, labelspacing=0.1)


def effect_overlap_panel(datasets, filename):

    cls_sep = ["10%", "25%", "40%"]
    
    fig, axes = plt.subplots(1, 3, figsize=plot_utils.set_fig_size(430, fraction=1.8, subplots=(1, 2.5)))
    fig.subplots_adjust(wspace=0, hspace=0) 
    fig.patch.set_linewidth(3)

    for row_num, score in enumerate(["fbeta_score"]):
        for col_num, noise in enumerate([0.1, 0.25, 0.4]):

            outer_axis = axes[col_num]
            outer_axis.patch.set_linewidth(4)

            remove_outer_spines(outer_axis, row_num, col_num)
            remove_lower_spines(outer_axis)
            remove_ticks(outer_axis)

            inner_axis = inset_axes(outer_axis, width=2, height=1.2, loc="center")

            if col_num == 0:
                outer_axis.set_ylabel(r"F$_2$ score", fontdict={"fontsize": 12})

            if row_num == 0:
                outer_axis.set_title(f"Label noise {cls_sep[col_num]}", pad=10)

            for i, (key, data) in enumerate(datasets.items()):
                _plot_groupwise(data.where(data.loc[:, "noise"] == noise).dropna(how="all"), "overlap", key, inner_axis, score, i)

                inner_axis.set_xticks(np.linspace(-2, 0, 6))
                inner_axis.set_xticklabels(np.round((np.linspace(-2, 0, 6) + 2) / 4, 1))
    
            plot_utils.format_axis(inner_axis, fig, ylim=[-0.05, 1.05], grid=True, xlabel="Class separability")
    
    fig.savefig(f"{filename}/panel_effect_overlap.pdf", edgecolor="k", transparent=True, bbox_inches="tight") 


def remove_lower_spines(axis):
    axis.spines["bottom"].set_visible(False)


def remove_spines(axis):

    for side in ['bottom','right','top','left']:
        axis.spines[side].set_visible(False)


def remove_outer_spines(axis, row_num, col_num, drop_lower=False):

    if drop_lower:
        axis.spines["bottom"].set_visible(False)

    # Upper left 
    if row_num == 0 and col_num == 0:
        axis.spines["top"].set_visible(False)
        axis.spines["left"].set_visible(False)

    # Upper right 
    if row_num == 0 and col_num == 2:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    # Lower left 
    if row_num == 2 and col_num == 0:
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)

    # Lower right 
    if row_num == 2 and col_num == 2:
        axis.spines["bottom"].set_visible(False)
        axis.spines["right"].set_visible(False)

    # Top 
    if row_num == 0 and col_num > 0:
        axis.spines["top"].set_visible(False)

    # Bottom 
    if row_num == 2 and col_num > 0:
        axis.spines["bottom"].set_visible(False)

    # Left edge 
    if row_num == 1 and col_num == 0:
        axis.spines["left"].set_visible(False)

    # Rght edge 
    if row_num == 1 and col_num == 2:
        axis.spines["right"].set_visible(False)


def effect_margin_losses(datasets, filename):

    cls_sep = ["100%", "75%", "50%"]
    experiments = [("delta", "delta_adaptive"), ("lmdnn", "lmdnn_adaptive")]
    
    fig, axes = plt.subplots(2, 3, figsize=plot_utils.set_fig_size(430, fraction=1.8, subplots=(2, 2.5)))
    fig.subplots_adjust(wspace=0, hspace=0) 
    fig.patch.set_linewidth(3)
    
    for row_num, experiment in enumerate(experiments):
        for col_num, overlap in enumerate([-2, -1, 0]):

            outer_axis = axes[row_num, col_num]
            outer_axis.patch.set_linewidth(4)

            remove_outer_spines(outer_axis, row_num, col_num)
            remove_ticks(outer_axis)

            inner_axis = inset_axes(outer_axis, width=2, height=1.2, loc="center")

            if col_num == 0:
                outer_axis.set_ylabel(r"F$_2$ score", fontdict={"fontsize": 12})

            if row_num == 0:
                outer_axis.set_title(f"Class separability {cls_sep[col_num]}", pad=10)

            for i, key in enumerate(experiment):
                data = datasets[key]
                _plot_groupwise(data.where(data.loc[:, "overlap"] == overlap).dropna(how="all"), "noise", key, inner_axis, "fbeta_score", i)

            plot_utils.format_axis(inner_axis, fig, ylim=[0.2, 1.05], grid=True, xlabel="Label noise")
    
    fig.savefig(f"{filename}/effect_margin_losses.pdf", edgecolor="k", transparent=True, bbox_inches="tight") 


def performance_curves(datasets, filename):

    column_lables = {"delta": "Delta", "focal": "Focal", "lmdnn": "Lmdnn"}

    fig, axes = plt.subplots(2, 3, figsize=plot_utils.set_fig_size(430, fraction=2.5, subplots=(2, 3.5)))
    fig.subplots_adjust(wspace=0, hspace=0) 
    fig.patch.set_linewidth(3)
    
    for col_num, (label, data) in enumerate(datasets.items()):
        for row_num, ci in enumerate([0, 1]):

            outer_axis = axes[row_num, col_num]
            outer_axis.patch.set_linewidth(4)

            remove_outer_spines(outer_axis, row_num, col_num, drop_lower=row_num==1)
            remove_ticks(outer_axis)

            inner_axis = inset_axes(outer_axis, width=2.5, height=1, loc="center")
            #inner_axis = inset_axes(outer_axis, width=2.5, height=1.2, loc="center")

            if col_num == 0:
                outer_axis.set_ylabel("Predicting normal" if ci == 0 else "Predicting high grade", fontsize=12)

            if row_num == 0:
                outer_axis.set_title(column_lables[label], pad=0, fontsize=12)

            _plot_score_curve(data.loc[:, "y_true"].values, data.loc[:, "p_hat"].values, ci, inner_axis)

            plot_utils.format_axis(inner_axis, fig, xlim=(-1.01, 1.05), ylim=(0, 1.15), xlabel=r"Classification threshold, $\tau$", 
                                   ylabel=r"Sample coverage, $p_\tau$", grid=True)

    fig.savefig(f"{filename}/performance_curve.pdf", edgecolor="k", transparent=True, bbox_inches="tight")


def _plot_score_curve(y_true, p_hat, ci, axis):

    thresholds = np.linspace(-1, 1, 250)

    scores_ci = delta_score(y_true[y_true == ci], p_hat[y_true == ci]) 

    correct_clfs = np.ones(thresholds.size) * np.nan 
    for i, tau in enumerate(thresholds):
        correct_clfs[i] = sum(scores_ci <= tau) / (scores_ci.shape[0] + 1e-16)

    auc_left = auc(thresholds[thresholds <= 0], correct_clfs[thresholds <= 0])
    auc_right = auc(thresholds[thresholds > 0], correct_clfs[thresholds > 0])

    axis.annotate("AUC = " + "{:.3f}".format(auc_left), xy=(-0.8, 1.05), fontsize=9)
    axis.annotate("", xy=(-1.0, 1.025), xycoords='data', xytext=(0.0, 1.025), textcoords='data', 
                  arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color="dimgray", alpha=0.8))
    
    axis.plot(thresholds, correct_clfs, c=f"C{ci}")
    axis.axvline(x=0, ymin=0, ymax=0.9, color="dimgray", alpha=0.5, linestyle="--")
    axis.fill_between(thresholds, correct_clfs, color=f"C{ci}", alpha=0.3)
                        