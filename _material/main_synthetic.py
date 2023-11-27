import json
from collections import defaultdict 

import numpy as np 
import pandas as pd 
import tensorflow as tf 

from sklearn.metrics import matthews_corrcoef, confusion_matrix, fbeta_score

from utils import synthetic_data, delta_auc_score, delta_score 
from models import logreg
from losses import DeltaLoss, LMDNNLoss, LDAMLoss
from training_test_synthetic import train_predict_synthetic

from plotting.panels import effect_noise_panel, effect_overlap_panel, effect_margin_losses, performance_curves
from plotting.training import (plot_loss, plot_weights, plot_rates, 
                               plot_weight_outcomes, plot_beta_outcomes)
from plotting.imbalance import plot_scores, label_noise_on_class_ratio
from plotting.decision_surface import plot_decision_surface, plot_probability_surface
from plotting.noise import plot_overlap_noise_scores 
from plotting.data import plot_data, plot_datasets
from plotting.delta_plots import plot_score_curve

BASE_EXP = ""
BASE_RESULTS = ""

np.random.seed(42)
tf.random.set_seed(42)


def effective_number_weights(y_true, p_hat, epoch, alpha, beta=0.999):

    class_counts = np.bincount(y_true)

    scores = (1.0 - beta) / (1.0 - np.power(beta, class_counts))
    scores = scores / np.sum(scores) * len(class_counts)

    weights = np.zeros_like(p_hat)
    weights[y_true == 0] = scores[0]
    weights[y_true == 1] = scores[1]

    return weights


def focal_weights(y_true, p_pred, gamma=2):

    p_pred = np.squeeze(p_pred)
    p_target = y_true * p_pred + (1 - y_true) * (1 - p_pred)

    neg, pos = np.bincount(y_true.astype(np.int32))
    weight_0 = (1 / neg) * (y_true.size / 2.0)
    weight_1 = (1 / pos) * (y_true.size / 2.0)

    alpha = (y_true * weight_1) + ((1 - y_true) * weight_0)

    return alpha * np.power(1.0 - p_target, gamma)
    

def uniform_weights(y_true, p_hat):
    return np.ones(p_hat.size, dtype=np.float32)


def save_metadata(results, filename):

    meta = {
        "weighted_train_loss": results["weighted_train_loss"],
        "weighted_val_loss": results["weighted_val_loss"],
        "train_loss": results["train_loss"],
        "val_loss": results["val_loss"],
        "w_tilde": results["w_tilde"],
        "w_delta": results["w_delta"],
        "epochs": results["epochs"]
    }
    pd.DataFrame(meta).to_csv(f"{filename}_meta.csv")


def plot_results(output, model, path_to_fig):

    results = output["results"]

    plot_beta_outcomes(output["betas"], results, path_to_fig, fname="weight_updates")
    plot_weight_outcomes(output["weights"], results, path_to_fig, fname="sample_weights")
    plot_weight_outcomes(output["weights_tilde"], results, path_to_fig, fname="smooth_sample_weights", smooth=True)

    plot_decision_surface(f"{path_to_fig}_train", output["X_train"], results["y_true_train"], model)
    plot_probability_surface(f"{path_to_fig}_train", output["X_train"], results["y_true_train"], model)

    plot_decision_surface(f"{path_to_fig}_test", output["X_test"], results["y_true_test"], model)
    plot_probability_surface(f"{path_to_fig}_test", output["X_test"], results["y_true_test"], model)
                                
    plot_weights(results["epochs"], results["w_tilde_c0"], results["w_tilde_c1"], 
                 results["w_delta_c1"], results["w_delta_c1"], filename=path_to_fig)
                        
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"], filename=path_to_fig)

    plot_score_curve(results["y_true_test"], p_hat=results["p_hat_test"], filename=path_to_fig)


def _log_scores(results, output):

    _output = output["results"]

    tn, fp, fn, tp = confusion_matrix(_output["y_true_test"], _output["y_hat_test"], labels=[0, 1]).ravel()
    results["tnr"].append(tn / (tn + fp))
    results["tpr"].append(tp / (tp + fn))
    results["acc_bal"].append(0.5 * (results["tnr"][-1] + results["tpr"][-1]))

    results["mcc"].append(matthews_corrcoef(_output["y_true_test"], _output["y_hat_test"]))
    results["f1_score"].append(fbeta_score(_output["y_true_test"], _output["y_hat_test"], beta=1))
    results["fbeta_score"].append(fbeta_score(_output["y_true_test"], _output["y_hat_test"], beta=2))


def effect_smooth_weights_plot(results_dir, overlaps=None, noise=None, abort=True, ratio=0.05):

    datasets, setup_labels = [], []
    
    weights_name = ["uniform", "focal", "focal"]
    losses_name = ["bce", "bce", "bce"]
    for i, exp_id in enumerate(["unweighted", "focal", "focal_adaptive"]):

        path_to_results = f"{BASE_RESULTS}/loss_comparison/{results_dir}/{exp_id}"
        fname = f"logreg_{weights_name[i]}_{losses_name[i]}"
        
        data = pd.read_csv(f"{BASE_RESULTS}/loss_comparison/{results_dir}/{exp_id}/{fname}_results.csv", index_col=0)

        datasets.append(data)
        setup_labels.append(exp_id)

    if noise is not None:

        for noise_level in noise:

            plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                    target="fbeta_score", ratio=ratio, noise=noise_level, overlap=None, labels=setup_labels)
            plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                    target="tnr", ratio=ratio, noise=noise_level, overlap=None, labels=setup_labels)
            plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                    target="tpr", ratio=ratio, noise=noise_level, overlap=None, labels=setup_labels)
    
    else:    
        for overlap in overlaps:

            plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                    target="fbeta_score", ratio=ratio, noise=noise, overlap=overlap, labels=setup_labels)
            plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                    target="tnr", ratio=ratio, noise=noise, overlap=overlap, labels=setup_labels)
            plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                    target="tpr", ratio=ratio, noise=noise, overlap=overlap, labels=setup_labels)
        
    if abort:
        assert False 


def _print_score_table(results, experiment):

    print("-" * 80)
    print("Experiment:", experiment)
    for ratio, ratio_frame in results.groupby("imbalance"):
        print("ratio:", ratio)
        print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format("overlap", "noise", 
                                                                        "tnr", "tnr_std", "tpr", "tpr_std", 
                                                                        "fscore_avg", "fscore_std", "f1", "mcc"))
        print("-" * 80)
        for overlap, overlap_frame in ratio_frame.groupby("overlap"):
            
            for noise, noise_frame in overlap_frame.groupby("noise"):
                print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
                    overlap, noise, 
                    np.round(noise_frame.loc[:, "tnr"].mean(), 3),
                    np.round(noise_frame.loc[:, "tnr"].std(), 3),
                    np.round(noise_frame.loc[:, "tpr"].mean(), 3),
                    np.round(noise_frame.loc[:, "tpr"].std(), 3),
                    np.round(noise_frame.loc[:, "fbeta_score"].mean(), 3),
                    np.round(noise_frame.loc[:, "fbeta_score"].std(), 3),
                    np.round(noise_frame.loc[:, "f1_score"].std(), 3),
                    np.round(noise_frame.loc[:, "mcc"].std(), 3)))

        print("-" * 80)


def plot_effect_noise(results_dir):

    datasets = {}

    weights_name = ["uniform", "focal", "focal"]
    losses_name = ["bce", "bce", "bce"]
    for i, exp_id in enumerate(["unweighted", "focal", "focal_adaptive"]):
    
        path_to_results = f"{BASE_RESULTS}/loss_comparison/synthetic/{results_dir}/{exp_id}"
        fname = f"logreg_{weights_name[i]}_{losses_name[i]}"
        
        datasets[exp_id] = pd.read_csv(f"{path_to_results}/{fname}_results.csv", index_col=0)

    effect_noise_panel(datasets, f"{BASE_RESULTS}/loss_comparison/synthetic/{results_dir}")
    

def plot_effect_overlap(results_dir):

    datasets = {}

    weights_name = ["focal", "focal"]
    losses_name = ["bce", "delta"]
    for i, exp_id in enumerate(["focal", "delta"]):
    
        path_to_results = f"{BASE_RESULTS}/loss_comparison/synthetic/{results_dir}/{exp_id}"
        fname = f"logreg_{weights_name[i]}_{losses_name[i]}"
        
        datasets[exp_id] = pd.read_csv(f"{path_to_results}/{fname}_results.csv", index_col=0)

    effect_overlap_panel(datasets, f"{BASE_RESULTS}/loss_comparison/synthetic/{results_dir}")


def plot_effect_margin_losses(results_dir):

    datasets = {}

    weights_name = ["focal", "focal", "focal", "focal"]
    losses_name = ["lmdnn", "lmdnn", "delta", "delta"]
    for i, exp_id in enumerate(["lmdnn", "lmdnn_adaptive", "delta", "delta_adaptive"]):
    
        path_to_results = f"{BASE_RESULTS}/loss_comparison/synthetic/{results_dir}/{exp_id}"
        fname = f"logreg_{weights_name[i]}_{losses_name[i]}"
        
        datasets[exp_id] = pd.read_csv(f"{path_to_results}/{fname}_results.csv", index_col=0)

    effect_margin_losses(datasets, f"{BASE_RESULTS}/loss_comparison/synthetic/{results_dir}")


def plot_performance_curves(results_dir):

    datasets = {}
    for i, exp_id in enumerate(["focal", "delta", "lmdnn"]):    
        datasets[exp_id] = pd.read_csv(f"{BASE_RESULTS}/loss_comparison/{results_dir}/{exp_id}/predicted.csv", index_col=0)

    performance_curves(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}") 


def main():

    loss_fn = {
        "bce": tf.keras.losses.BinaryCrossentropy(),
        "delta": DeltaLoss(),
        "lmdnn": LMDNNLoss(C=1)
    }

    model_fn = {
        "logreg": logreg
    }

    weights_fn = {
        "uniform": uniform_weights,
        "focal": focal_weights,
        "effect": effective_number_weights
    }
    
    effect_smooth_weights_plot("synthetic/comparing_methods", [-2, -1, 0])  
    #effect_smooth_weights_plot("synthetic/overlap", noise=[0.1, 0.25, 0.4])  
    
    results_dir = "synthetic"
    source = "overlap"

    with open(f"{BASE_EXP}/simulations/{source}.json", "r") as json_file:
        experiments = json.load(json_file)

    filenames = []   
    for exp_id in ["focal", "focal_adaptive", "delta", "delta_adaptive", "lmdnn", "lmdnn_adaptive"]: 
    
        print("Experiment:", exp_id)
        path_to_results = f"{BASE_RESULTS}/loss_comparison/{results_dir}/{source}/{exp_id}"
        
        settings = experiments[exp_id]
        for model_name, learning_rate in zip(settings["models"], settings["learning_rates"]):            

            train_configs = zip(settings["weights"], settings["smooth_weights"], settings["adaptive_weights"])
            for weights_name, smooth_weights, adaptive_weights in train_configs:
                for loss_name in settings["losses"]:

                    results = defaultdict(list)

                    fname = f"{model_name}_{weights_name}_{loss_name}"
                    filenames.append(f"{path_to_results}/{fname}")

                    for imbalance in settings["fractions"]:
                        for noise, overlap in zip(settings["noise"], settings["overlap"]):
                            for seed in settings["random_seeds"]:

                                filename = f"{path_to_results}/{model_name}_{weights_name}_{loss_name}"
                                filename += f"r{imbalance}_n{noise}_o{overlap}_s{seed}"
                        
                                X, y = synthetic_data(imbalance=imbalance, overlap=overlap, noise=noise, seed=seed,
                                                      keep_balance=bool(experiments["keep_balance"]))
                            
                                model, output = train_predict_synthetic(X, y,  model_fn[model_name], loss_fn[loss_name],
                                                                        sample_weights_fn=weights_fn[weights_name],
                                                                        learning_rate=learning_rate,
                                                                        smooth_weights=bool(smooth_weights),
                                                                        adaptive_weights=bool(adaptive_weights), 
                                                                        max_epochs=experiments["max_epochs"],
                                                                        pre_train=experiments["pre_train"],
                                                                        seed=seed)
                                
                                _log_scores(results, output)
                                
                                results["seed"].append(seed)
                                results["noise"].append(noise)
                                results["overlap"].append(overlap)
                                results["imbalance"].append(imbalance)
        
                                plot_results(output, model, filename)

                    pd.DataFrame(results).to_csv(f"{filenames[-1]}_results.csv")

    
if __name__ == "__main__":
    main()