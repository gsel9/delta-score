import json
from collections import defaultdict 

import numpy as np 
import pandas as pd 
import tensorflow as tf 

from sklearn.metrics import (matthews_corrcoef, confusion_matrix, 
                             recall_score, precision_score, fbeta_score)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from utils import synthetic_data, delta_auc_score, delta_score
from models import logreg_multi
from utils_multiclass import synthetic_data_multiclass, delta_auc_score_multiclass
from losses_multiclass import DeltaLossMulticlass, LMDNNLossMulticlass
from training_test_synthetic import train_predict_synthetic

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


def focal_weights_multiclass(y_true, p_pred, gamma=2):

    _, counts = np.unique(np.argmax(y_true, axis=1), return_counts=True)
    scales = (1 / counts) * (y_true.shape[0] / y_true.shape[1])

    return np.sum(y_true * scales, axis=1) * np.power(1.0 - np.sum(y_true * p_pred, axis=1), gamma)
        

def uniform_weights_multiclass(y_true, p_hat):
    return np.ones(y_true.shape[0])


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
    
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"], filename=path_to_fig)
    

def _log_scores(results, output, labels=[0, 1, 2]):

    _output = output["results"]

    cmat = confusion_matrix(_output["y_true_test"], _output["y_hat_test"], labels=labels)

    fp = cmat.sum(axis=0) - np.diag(cmat)  
    fn = cmat.sum(axis=1) - np.diag(cmat)
    tp = np.diag(cmat)
    tn = cmat.sum() - (fp + fn + tp)

    tnr = tn / (tn + fp) 
    tpr = tp / (tp + fn)

    for i in labels:

        results[f"tnr_c{i}"].append(tnr[i])
        results[f"tpr_c{i}"].append(tpr[i])
        results[f"fbeta_score_c{i}"].append(fbeta_score(_output["y_true_test"] == i, _output["y_hat_test"] == i, beta=2))

    results["mcc"].append(matthews_corrcoef(_output["y_true_test"], _output["y_hat_test"]))

    class_weight = compute_class_weight("balanced", labels, _output["y_true_test"])

    results[f"tnr_mean"].append(np.average(tnr, weights=class_weight))
    results[f"tpr_mean"].append(np.average(tpr, weights=class_weight))

    score = fbeta_score(_output["y_true_test"], _output["y_hat_test"], beta=2, average=None)
    results["fbeta_score_avg"].append(np.average(score, weights=class_weight))
    

def _print_score_table(results, experiment, noise=None):

    if noise is not None:
        results = results.where(results.loc[:, "noise"] == noise).dropna(how="all")

    print("-" * 80)
    print("Experiment:", experiment)
    for ratio, ratio_frame in results.groupby("imbalance"):

        print("ratio:", ratio)
        print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format("overlap", "noise", 
                                                                       "tnr", "tnr_std", "tpr", "tpr_std", 
                                                                       "f2_score_avg", "f2_score_std"))
        print("-" * 80)
        for overlap, overlap_frame in ratio_frame.groupby("overlap"):
            for noise, noise_frame in overlap_frame.groupby("noise"):

                print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
                    overlap, noise, 
                    np.round(noise_frame.loc[:, "tnr_mean"].mean(), 3), 
                    np.round(noise_frame.loc[:, "tnr_mean"].std(), 3),
                    np.round(noise_frame.loc[:, "tpr_mean"].mean(), 3), 
                    np.round(noise_frame.loc[:, "tpr_mean"].std(), 3),
                    np.round(noise_frame.loc[:, "fbeta_score_avg"].mean(), 3), 
                    np.round(noise_frame.loc[:, "fbeta_score_avg"].std(), 3)))
        print("-" * 80)


def effect_smooth_weights_plot(results_dir, overlaps, abort=True):

    datasets, setup_labels = [], []
    
    weights_name = ["focal", "focal"]
    losses_name = ["bce", "bce"]
    for i, exp_id in enumerate(["focal", "focal_adaptive"]):

    #weights_name = ["focal", "focal"]
    #losses_name = ["lmdnn", "lmdnn"]
    #for i, exp_id in enumerate(["lmdnn", "lmdnn_adaptive"]):

    #weights_name = ["focal", "focal"]
    #losses_name = ["delta", "delta"]
    #for i, exp_id in enumerate(["delta", "delta_adaptive"]):

        path_to_results = f"{BASE_RESULTS}/loss_comparison/{results_dir}/{exp_id}"
        fname = f"logreg_{weights_name[i]}_{losses_name[i]}"
        
        data = pd.read_csv(f"{BASE_RESULTS}/loss_comparison/{results_dir}/{exp_id}/{fname}_results.csv", index_col=0)

        datasets.append(data)
        setup_labels.append(exp_id)

    for overlap in overlaps:

        plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                target="fbeta_score_avg", ratio=0.05, overlap=overlap, labels=setup_labels)
        plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                target="tnr_mean", ratio=0.05, overlap=overlap, labels=setup_labels)
        plot_overlap_noise_scores(datasets, f"{BASE_RESULTS}/loss_comparison/{results_dir}", 
                                target="tpr_mean", ratio=0.05, overlap=overlap, labels=setup_labels)
    
    if abort:
        assert False 


def main():
    
    loss_fn = {
        "bce": tf.keras.losses.CategoricalCrossentropy(),
        "delta": DeltaLossMulticlass(),
        "lmdnn": LMDNNLossMulticlass(C=1)
    }

    model_fn = {
        "logreg": logreg_multi
    }

    weights_fn = {
        "uniform": uniform_weights_multiclass,
        "focal": focal_weights_multiclass
    }

    source = "overlap"
    results_dir = "synthetic_multiclass"

    with open(f"{BASE_EXP}/simulations/{source}.json", "r") as json_file:
        experiments = json.load(json_file)

    filenames = []
    for exp_id in ["focal_adaptive", "focal"]:
    #for exp_id in ["delta", "delta_adaptive"]: 
    #for exp_id in ["lmdnn", "lmdnn_adaptive"]: 
        
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

                                X, y = synthetic_data_multiclass(n_features=5, n_redundant=2,
                                                                 imbalance=imbalance, overlap=overlap, noise=noise, seed=seed,
                                                                 keep_balance=bool(experiments["keep_balance"]), ohe_y=True)
   
                                model, output = train_predict_synthetic(X, y,  model_fn[model_name], loss_fn[loss_name],
                                                                        sample_weights_fn=weights_fn[weights_name],
                                                                        learning_rate=learning_rate,
                                                                        smooth_weights=False,
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