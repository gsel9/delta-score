import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

from . import plot_utils 

plot_utils.setup()

ylabels = {
    "c0": "TNR",
    "c1": "TPR",
    "f": r"F$_{2}$"
}

ylims = {
    "c0": [0, 1], 
    "c1": [0, 1],
    "f": [0, 1] 
}


def label_noise_on_class_ratio(settings, data_gen, path_to_fig, keep_balance=False, fname=""):

    for overlap in settings["overlap"]:

        mean_fractions = []
        for noise in settings["noise"]:

            fractions = []
            for seed in settings["random_seeds"]:
                
                _, y = data_gen(imbalance=settings["fraction"], overlap=overlap, noise=noise, seed=seed, keep_balance=keep_balance)
                fractions.append(sum(y) / y.size)

            mean_fractions.append(np.mean(fractions))

        fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.8, subplots=(1.1, 1)))
        axis.bar(settings["noise"], mean_fractions, width=0.025, fc="C0", alpha=0.5, ec="C0", lw=3)

        plot_utils.format_axis(axis, fig, xlabel="Label noise", ylabel="Class ratio", 
                               grid=True, axis_title=f"Class separatilibty: {1 - overlap}")

        axis.set_xticks(settings["noise"])
        axis.set_xticklabels(settings["noise"])
                    
        fig.tight_layout()
        if keep_balance:
            fig.savefig(f"{path_to_fig}/class_ratios_overlap{overlap}_balanced.pdf", transparent=True, bbox_inches="tight") 
        else:
            fig.savefig(f"{path_to_fig}/class_ratios_overlap{overlap}.pdf", transparent=True, bbox_inches="tight") 


def _load_scores(path_to_results, loss_name, model_name, scores):
    return pd.read_csv(f"{path_to_results}/{loss_name}_{model_name}_{scores}.csv", index_col=0)


def plot_scores(filenames, target, path_to_fig, fractions, labels=None, ylabel=None, xlabel=None): 

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.8, subplots=(1.1, 1)))

    for i, filename in enumerate(filenames):

        values = pd.read_csv(f"{filename}_{target}.csv", index_col=0)

        scores_avg = np.mean(values, axis=0).values[::-1]
        scores_std = np.std(values, axis=0).values[::-1]

        axis.plot(fractions[::-1], scores_avg, "-o", c=f"C{i}", label=labels[i] if labels is not None else str(i), 
                  markersize=4, linewidth=1, alpha=0.8)
        
    axis.set_xticks(fractions)
    axis.set_xticklabels(fractions)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True)

    plot_utils.format_axis(axis, fig, grid=True, ylabel=ylabel, ylim=[0, 1], xlabel=r"Class ratio") 
                
    fig.tight_layout()
    fig.savefig(f"{path_to_fig}/scores_{target}.pdf", transparent=True, bbox_inches="tight") 


def delta_score(y_true, p_pred):

    y_true = y_true.astype(int)
    p_pred = np.transpose(np.vstack([1.0 - p_pred, p_pred]))

    return p_pred[range(y_true.size), 1 - y_true] - p_pred[range(y_true.size), y_true] 
    

def plot_score_curve(y_true, p_hat=None, delta=None, path_to_fig=None, ylabel=None, fname=""):

    if ylabel is None:
        ylabel = "Fraction of samples"
    
    thresholds = np.linspace(-1, 1, 500)
    
    fig, axes = plt.subplots(2, 1, figsize=plot_utils.set_fig_size(430, fraction=0.8, subplots=(1.2, 1)))

    for ci, axis in enumerate(axes.ravel()):

        # Stratify per class 
        if delta is not None:
            scores_ci = delta[y_true == ci]
        else:
            scores_ci = delta_score(y_true[y_true == ci], p_hat[y_true == ci]) 

        correct_clfs = np.ones(thresholds.size) * np.nan 
        for i, tau in enumerate(thresholds):
            correct_clfs[i] = sum(scores_ci <= tau) / scores_ci.shape[0]

        auc_left = auc(thresholds[thresholds <= 0], correct_clfs[thresholds <= 0])
        auc_right = auc(thresholds[thresholds > 0], correct_clfs[thresholds > 0])

        axis.annotate(r"$A_{\delta} = $" + "{:.3f}".format(auc_left), xy=(-0.6, 1.05), fontsize=9)
        axis.annotate("", xy=(-1.0, 1.025), xycoords='data', xytext=(0.0, 1.025), textcoords='data', 
                      arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color="dimgray", alpha=0.8))
        
        axis.plot(thresholds, correct_clfs, c=f"C{ci}")
        axis.axvline(0.0, 0, 1, color="dimgray", alpha=0.5, linestyle="--")
        axis.fill_between(thresholds, correct_clfs, color=f"C{ci}", alpha=0.3)

        plot_utils.format_axis(axis, fig, xlim=(-1.01, 1.05), ylim=(0, 1.05), xlabel=r"$\tau$", ylabel=ylabel,
                               grid=True, axis_title=r"c$_{}$".format(ci))

    fig.tight_layout()
    fig.savefig(f"{path_to_fig}/delta_curve_{fname}.pdf", transparent=True, bbox_inches="tight") 


def _wasserstein_score(hist):   

    hist_ideal = np.zeros_like(hist)
    hist_ideal[0] = sum(hist)

    return np.round(wasserstein_distance(hist_ideal, hist), 3)


def plot_delta_score_histogram(y_true, p_hat=None, delta=None, path_to_fig=None, n_bins=50, fname=""):
    
    fig, axes = plt.subplots(2, 1, figsize=plot_utils.set_fig_size(430, fraction=0.8, subplots=(1.2, 1)))

    for ci, axis in enumerate(axes.ravel()):

        # Stratify per class 
        if delta is not None:
            scores_ci = delta[y_true == ci]
        else:
            scores_ci = delta_score(y_true[y_true == ci], p_hat[y_true == ci]) 

        hist, bins = np.histogram(scores_ci, bins=np.linspace(-1, 1, n_bins))
        
        axis.bar((bins[:-1] + bins[1:]) / 2, hist, width=0.01, fc=f"C{ci}", alpha=0.5, ec=f"C{ci}", lw=3)

        plot_utils.format_axis(axis, fig, xlim=(-1.0, 1.05), xlabel=r"$\delta$ score", ylabel="Sample count", 
                               grid=True, axis_title=r"c$_{}$".format(ci))

    fig.tight_layout()
    fig.savefig(f"{path_to_fig}/delta_histogram_{fname}.pdf", transparent=True, bbox_inches="tight") 