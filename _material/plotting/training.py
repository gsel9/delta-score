import numpy as np 
import matplotlib.pyplot as plt 

from . import plot_utils 

plot_utils.setup()


def plot_rates(epochs, eta_scores, filename, shift=0):

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))

    axis.plot(epochs[shift:], eta_scores[shift:], marker="o")

    if shift > 0:
        plot_utils.format_axis(axis, fig, ylim=plot_utils.set_ylim(eta_scores), xlabel=f"Epoch (last {abs(shift)})", ylabel=r"$\eta$", grid=True)
    else:
        plot_utils.format_axis(axis, fig, ylim=plot_utils.set_ylim(eta_scores), xlabel="Epoch", ylabel=r"$\eta$", grid=True)

    fig.legend()
    fig.tight_layout()
    fig.savefig(f"{filename}_rates.pdf", transparent=True, bbox_inches="tight") 


def plot_loss(epochs, train_loss, val_loss, filename, weighted=False, shift=0):

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))

    if weighted:
        axis.plot(epochs[shift:], train_loss[shift:], label="Training (weighted)", alpha=0.8)
        axis.plot(epochs[shift:], val_loss[shift:], label="Validation (weighted)", alpha=0.8)

    else:
        axis.plot(epochs[shift:], train_loss[shift:], label="Training", alpha=0.8)
        axis.plot(epochs[shift:], val_loss[shift:], label="Validation)", alpha=0.8)

    if shift > 0:
        plot_utils.format_axis(axis, fig, ylim=plot_utils.set_ylim(np.concatenate((train_loss, val_loss))), xlabel=f"Epoch (last {abs(shift)})", ylabel="Loss", grid=True)
    else:
        plot_utils.format_axis(axis, fig, ylim=plot_utils.set_ylim(np.concatenate((train_loss, val_loss))), xlabel="Epoch", ylabel="Loss", grid=True)

    fig.legend()
    fig.tight_layout()

    if weighted:
        fig.savefig(f"{filename}_weighted_loss.pdf", transparent=True, bbox_inches="tight") 
    else:
        fig.savefig(f"{filename}_loss.pdf", transparent=True, bbox_inches="tight") 


def plot_weights(epochs, w_tilde_c0, w_tilde_c1, w_delta_c0, w_delta_c1, filename, shift=0):

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))

    axis.plot(epochs[shift:], w_tilde_c0[shift:], label="w_tilde c0", marker="o", alpha=0.8)
    axis.plot(epochs[shift:], w_tilde_c1[shift:], label="w_tilde c1", marker="o", alpha=0.8)

    if shift > 0:
        plot_utils.format_axis(axis, fig, xlabel=f"Epoch (last {abs(shift)})", ylabel="Norm of weight set", grid=True)
    else:
        plot_utils.format_axis(axis, fig, xlabel="Epoch", ylabel="Norm of weight set", grid=True)
    
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"{filename}_weights.pdf", transparent=True, bbox_inches="tight") 


def plot_weight_samples(epochs, weights, filename, n_samples=20, ylabel=None, fname="sample_weights"):

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))

    n_samples = min(n_samples, weights.shape[0])
    for i in range(n_samples):
        axis.plot(epochs, weights[i], marker="o", linestyle="-", alpha=0.8)
    
    plot_utils.format_axis(axis, fig, xlabel="Epoch", grid=True,
                           ylabel="Sample weight" if ylabel is None else ylabel)
    
    fig.tight_layout()
    fig.savefig(f"{filename}_{fname}.pdf", transparent=True, bbox_inches="tight") 
     

def plot_beta_outcomes(betas, results, path_to_fig, fname="betas"):

    betas_c0_correct = betas[np.logical_and(results["y_true_train"] == 0, results["y_hat_train"] == 0)]
    betas_c1_correct = betas[np.logical_and(results["y_true_train"] == 1, results["y_hat_train"] == 1)]
    betas_c0_incorrect = betas[np.logical_and(results["y_true_train"] == 0, results["y_hat_train"] == 1)]
    betas_c1_incorrect = betas[np.logical_and(results["y_true_train"] == 1, results["y_hat_train"] == 0)]

    plot_weight_samples(results["epochs"], betas_c0_correct, path_to_fig, ylabel="Weight update (correct c0)", fname=f"{fname}_c0_correct")
    plot_weight_samples(results["epochs"], betas_c1_correct, path_to_fig, ylabel="Weight update (correct c1)", fname=f"{fname}_c1_correct")
    plot_weight_samples(results["epochs"], betas_c0_incorrect, path_to_fig, ylabel="Weight update (incorrect c0)", fname=f"{fname}_c0_incorrect")
    plot_weight_samples(results["epochs"], betas_c1_incorrect, path_to_fig, ylabel="Weight update (incorrect c1)", fname=f"{fname}_c1_incorrect")
        

def plot_weight_outcomes(weights, results, path_to_fig, fname="sampe_weights", smooth=False):

    weights_c0_correct = weights[np.logical_and(results["y_true_train"] == 0, results["y_hat_train"] == 0)]
    weights_c1_correct = weights[np.logical_and(results["y_true_train"] == 1, results["y_hat_train"] == 1)]
    weights_c0_incorrect = weights[np.logical_and(results["y_true_train"] == 0, results["y_hat_train"] == 1)]
    weights_c1_incorrect = weights[np.logical_and(results["y_true_train"] == 1, results["y_hat_train"] == 0)]

    if smooth:

        plot_weight_samples(results["epochs"], weights_c0_correct, path_to_fig, ylabel="Sample weight (correct c0)", fname=f"{fname}_c0_correct")
        plot_weight_samples(results["epochs"], weights_c1_correct, path_to_fig, ylabel="Sample weight (correct c1)", fname=f"{fname}_c1_correct")
        plot_weight_samples(results["epochs"], weights_c0_incorrect, path_to_fig, ylabel="Sample weight (failing on c0)", fname=f"{fname}_c0_incorrect")
        plot_weight_samples(results["epochs"], weights_c1_incorrect, path_to_fig, ylabel="Sample weight (failing on c1)", fname=f"{fname}_c1_incorrect")

    else:

        plot_weight_samples(results["epochs"], weights_c0_correct, path_to_fig, ylabel="Smooth sample weight (correct c0)", fname=f"{fname}_c0_correct")
        plot_weight_samples(results["epochs"], weights_c1_correct, path_to_fig, ylabel="Smooth sample weight (correct c1)", fname=f"{fname}_c1_correct")
        plot_weight_samples(results["epochs"], weights_c0_incorrect, path_to_fig, ylabel="Smooth sample weight (failing on c0)", fname=f"{fname}_c0_incorrect")
        plot_weight_samples(results["epochs"], weights_c1_incorrect, path_to_fig, ylabel="Smooth sample weight (failing on c1)", fname=f"{fname}_c1_incorrect")
