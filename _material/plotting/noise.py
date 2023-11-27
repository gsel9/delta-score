import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

from . import plot_utils 

plot_utils.setup()

ylabels = {
    "tnr": "TNR",
    "tpr": "TPR",
    "fbeta_score": r"F$_{2}$",
    "mcc": "MCC",
    "fbeta_score_avg": "F",
    "tnr_mean": "<",
    "tpr_mean": "T",
    "fbeta_score_c2": "fbeta_score_c2",
    "tnr_c2": "tnr_c2",
    "tpr_c2": "tpr_c2"
}

ylims = {
    "c0": [0, 1], 
    "c1": [0, 1], 
    "f": [0, 1] 
}

exp_names = {
    "unweighted_lowlr": "Uniform",
    "unweighted": "Uniform", 
    "static": "Static",
    "focal": "Focal", 
    "focal_lowlr": "Focal", 
    "focal_adaptive": "Adaptive",
    "focal_adaptive_lowlr_relu": "Adaptive relu",
    "focal_adaptive_lowlr_sigmoid": "Adaptive sigmoid",
    "focal_smooth": "Focal smooth", 
    "delta": "Delta", 
    "delta_adaptive": "Delta adaptive", 
    "lmdnn": "LMDNN",
    "lmdnn_adaptive": "LMDNN smooth"
}


def plot_label_noise_on_class_ratio(abort=True):

    with open(f"{BASE_EXP}/simulations/label_noise_on_class_ratio.json", "r") as json_file:
        experiments = json.load(json_file)
    
    label_noise_on_class_ratio(experiments, synthetic_data, f"{BASE_RESULTS}/loss_comparison/synthetic/label_noise_on_class_ratio", keep_balance=False)
    label_noise_on_class_ratio(experiments, synthetic_data, f"{BASE_RESULTS}/loss_comparison/synthetic/label_noise_on_class_ratio", keep_balance=True)
    
    if abort:
        assert False  


def plot_overlap_noise_scores(datasets, path_to_fig, target="", ratio=0, noise=None, overlap=0, labels=None, ylabel=None, xlabel=None): 

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.65, subplots=(1.15, 0.9)))

    for i, data in enumerate(datasets):
        
        filtered = data.where(data.loc[:, "imbalance"] == ratio).dropna(how="all")

        if noise is not None:
            filtered = filtered.where(filtered.loc[:, "noise"] == noise).dropna(how="all")
            groups = filtered.groupby("overlap")[target]

        else:
            filtered = filtered.where(filtered.loc[:, "overlap"] == overlap).dropna(how="all")
            groups = filtered.groupby("noise")[target]

        scores_avg = groups.mean()
        scores_std = groups.std()

        if noise is not None:
            xticks = np.round(((-1.0 * scores_avg.index.values) + 2) / 4, 2)
        else:
            xticks = scores_avg.index.values

        axis.plot(xticks, scores_avg.values, marker="o", linestyle="", c=f"C{i}", label=exp_names[labels[i]] if labels is not None else str(i), 
                  markersize=4, alpha=0.8)
        axis.plot(xticks, scores_avg.values, marker="", linestyle="-", c=f"C{i}", alpha=0.5)
        
        axis.fill_between(xticks, scores_avg.values - scores_std, scores_avg.values + scores_std, color=f"C{i}", alpha=0.1)
        axis.fill_between(xticks, scores_avg.values - scores_std, scores_avg.values + scores_std, color=f"C{i}", alpha=0.1)

    axis.set_xticks(xticks)
    axis.set_xticklabels(xticks)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True)
    
    ylim = [-0.05, 1.05]
    
    xlabel = "Label noise" if noise is None else "Class separability"
    plot_utils.format_axis(axis, fig, grid=True, ylabel=f"Performance score ({ylabels[target]})", ylim=ylim, xlabel=xlabel) 
                
    fig.tight_layout()
    if noise is not None:
        fig.savefig(f"{path_to_fig}/noise_scores_{target}_r{ratio}_n{noise}.pdf", transparent=True, bbox_inches="tight") 
    else:
        fig.savefig(f"{path_to_fig}/noise_scores_{target}_r{ratio}_o{overlap}.pdf", transparent=True, bbox_inches="tight") 
