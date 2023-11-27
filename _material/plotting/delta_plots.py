import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc
from scipy.stats import wasserstein_distance

from . import plot_utils 

plot_utils.setup()


def delta_score(y_true, p_pred):

    y_true = y_true.astype(int)
    p_pred = np.transpose(np.vstack([1.0 - p_pred, p_pred]))

    return p_pred[range(y_true.size), 1 - y_true] - p_pred[range(y_true.size), y_true] 
    

def plot_score_curve(y_true, p_hat=None, delta=None, filename=None, ylabel=None, classes=[0, 1]):

    if ylabel is None:
        ylabel = r"Sample coverage, $p_\tau$" 

    thresholds = np.linspace(-1, 1, 250)
    
    for ci in classes:

        # Stratify per class 
        if delta is not None:
            scores_ci = delta[y_true == ci]
        else:
            scores_ci = delta_score(y_true[y_true == ci], p_hat[y_true == ci]) 

        correct_clfs = np.ones(thresholds.size) * np.nan 
        for i, tau in enumerate(thresholds):
            correct_clfs[i] = sum(scores_ci <= tau) / (scores_ci.shape[0] + 1e-16)

        auc_left = auc(thresholds[thresholds <= 0], correct_clfs[thresholds <= 0])
        auc_right = auc(thresholds[thresholds > 0], correct_clfs[thresholds > 0])

        fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.8, subplots=(1, 1)))

        axis.annotate(f"AUC = " + "{:.3f}".format(auc_left), xy=(-0.6, 1.05), fontsize=9)
        axis.annotate("", xy=(-1.0, 1.025), xycoords='data', xytext=(0.0, 1.025), textcoords='data', 
                      arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color="dimgray", alpha=0.8))
        
        axis.plot(thresholds, correct_clfs, c=f"C{ci}")
        axis.axvline(0.0, 0, 1, color="dimgray", alpha=0.5, linestyle="--")
        axis.fill_between(thresholds, correct_clfs, color=f"C{ci}", alpha=0.3)

        axis_title = "Predicting normal" if ci == 0 else "Predicting high grade"
        plot_utils.format_axis(axis, fig, xlim=(-1.01, 1.05), ylim=(0, 1.15), xlabel=r"Error tolerance, $\tau$", ylabel=ylabel,
                               grid=True, axis_title=axis_title) 

        fig.tight_layout()
        fig.savefig(f"{filename}_delta_curve_c{ci}.pdf", transparent=True, bbox_inches="tight") 


def _wasserstein_score(hist):   

    hist_ideal = np.zeros_like(hist)
    hist_ideal[0] = sum(hist)

    return np.round(wasserstein_distance(hist_ideal, hist), 3)


def plot_delta_score_histogram(y_true, p_hat=None, delta=None, path_to_fig=None, n_bins=50, fname=""):
    
    for ci, axis in enumerate(axes.ravel()):

        fig, axes = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.8, subplots=(1, 1)))

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
        fig.savefig(f"{path_to_fig}/delta_histogram_c{ci}_{fname}.pdf", transparent=True, bbox_inches="tight") 