import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc
from scipy.stats import wasserstein_distance

from . import plot_utils 

plot_utils.setup()


def plot_data(filename, X, y):

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.9, subplots=(1, 1)))

    labels = np.unique(y)
    for label in labels:
        axis.plot(X[y == label, 0], X[y == label, 1], marker="o", linestyle="", alpha=0.7, label=f"C{int(label)}")

    plot_utils.format_axis(axis, fig, xlabel="x1", ylabel="x2", grid=True)

    fig.legend()
    fig.tight_layout()
    fig.savefig(f"{filename}_data.pdf", transparent=True, bbox_inches="tight") 


def plot_datasets(data_generator, abort=True):

    plot_data(f"{BASE_RESULTS}/loss_comparison/synthetic_non_fixed/basic_sample", 
              *data_generator(n_samples=200, imbalance=0.05, overlap=0, noise=0, seed=24, keep_balance=False, n_features=2))
    plot_data(f"{BASE_RESULTS}/loss_comparison/synthetic_non_fixed/noise_sample", 
              *data_generator(n_samples=200, imbalance=0.05, overlap=0, noise=0.4, seed=24, keep_balance=False, n_features=2))
    plot_data(f"{BASE_RESULTS}/loss_comparison/synthetic_non_fixed/overlap_sample", 
              *data_generator(n_samples=200, imbalance=0.05, overlap=0.5, noise=0, seed=24, keep_balance=False, n_features=2))
    plot_data(f"{BASE_RESULTS}/loss_comparison/synthetic_non_fixed/overlap_noise_sample", 
              *data_generator(n_samples=200, imbalance=0.05, overlap=0.5, noise=0.4, seed=24, keep_balance=False, n_features=2))

    if abort:
        assert False