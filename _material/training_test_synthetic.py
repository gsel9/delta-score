import math 
from collections import defaultdict

import numpy as np 
import pandas as pd 
import tensorflow as tf

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K 

from utils import (delta_score, validation_split, delta_auc_score, update_alpha,
                   DecaySchedule, AlphaSchedule, ConvergenceMonitor, prediction_results)
from utils_multiclass import multiclass_prediction_results

np.random.seed(42)
tf.random.set_seed(42)


def scale_train_test_val(X_train, X_test=None, X_val=None):

    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_std = scaler.transform(X_val)

    if X_test is not None:
        X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, X_val_std


def learning_rate_scheduler(init_lr, pre_train):

    def _learning_rate_scheduler(epoch, learning_rate):
        
        # Adjust for potential pre-training 
        epoch = epoch - pre_train 

        if epoch > 20:
            return init_lr / (1 + epoch / 2)

        return init_lr
    
    return _learning_rate_scheduler


def _track_loss_values(X_train, X_val, y_train, y_val, model, w_tilde_train, w_tilde_val, output, loss_fn):

    if np.ndim(y_train) < 2:
        y_train = y_train[:, None]
    
    if np.ndim(y_val) < 2:
        y_val = y_val[:, None]

    output["weighted_train_loss"].append(float(loss_fn(y_train, model(X_train), sample_weight=w_tilde_train[:, None])))
    output["weighted_val_loss"].append(float(loss_fn(y_val, model(X_val), sample_weight=w_tilde_val[:, None])))
    output["train_loss"].append(float(loss_fn(y_train, model(X_train))))
    output["val_loss"].append(float(loss_fn(y_val, model(X_val))))


def _train_synthetic(X_train, X_val, y_train, y_val, model_fn, loss_fn, learning_rate, sample_weights_fn, lr_decay=False,
                    smooth_weights=False, verbose=0, epochs_per_update=1, max_epochs=100, batch_size=64, pre_train=0,
                    seed=42, adaptive_weights=False):

    model = model_fn(n_features=X_train.shape[1], y=y_train, seed=seed)
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=0.0001, momentum=0.9), loss=loss_fn)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.0001, momentum=0.9), loss=loss_fn)
    
    if pre_train > 0:
        
        print(f"-*- Pre-training for max {pre_train} epochs -*-")
        history = model.fit(x=X_train, y=y_train, batch_size=batch_size, verbose=verbose, epochs=pre_train,
                            validation_data=(X_val, y_val), 
                            sample_weight=compute_sample_weight("balanced", y_train),
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)])

        pre_train = len(history.history["loss"])

    callbacks = []
    if lr_decay:
        callbacks += [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)]
    
    cmonitor = ConvergenceMonitor(patience=50, min_delta=1e-5, early_stopping=True)

    amonitor_val = AlphaSchedule(sample_size=y_val.shape[0], method="relu")
    amonitor_train = AlphaSchedule(sample_size=y_train.shape[0], method="relu")

    alphas, weights, weights_tilde = [], [], []
    output = defaultdict(list)

    # Initialize 
    w_tilde_train = sample_weights_fn(y_train, np.squeeze(model(X_train).numpy()))    
    w_tilde_val = sample_weights_fn(y_val, np.squeeze(model(X_val).numpy()))   

    max_epochs = max_epochs - pre_train
    for epoch in range(pre_train, max_epochs, epochs_per_update):

        # Resume training 
        model.fit(X_train, y_train, sample_weight=w_tilde_train, verbose=verbose, callbacks=callbacks,
                epochs=epoch + epochs_per_update, initial_epoch=epoch, batch_size=batch_size,
                validation_data=(X_val, y_val))

        w_train = sample_weights_fn(y_train, np.squeeze(model(X_train).numpy()))    
        w_val = sample_weights_fn(y_val, np.squeeze(model(X_val).numpy()))   
        
        if cmonitor.should_stop(float(loss_fn(y_val, np.squeeze(model(X_val)))), model):
            print(f"Early stopping after {epoch // epochs_per_update + 1} iterations and total epochs {epoch + epochs_per_update}")
            break 
        
        _track_loss_values(X_train, X_val, y_train, y_val, model, w_tilde_train, w_tilde_val, output, loss_fn)
        
        update_step = (epoch - pre_train + epochs_per_update) // epochs_per_update

        if adaptive_weights:

            alpha_train = amonitor_train.estimate(np.squeeze(model(X_train).numpy()), y_train) 
            alpha_val = amonitor_val.estimate(np.squeeze(model(X_val).numpy()), y_val)

        else:
            alpha_train = np.ones_like(w_train)
            alpha_val = np.ones_like(w_val)
            
        w_tilde_train = np.squeeze((1 - alpha_train) * w_tilde_train + alpha_train * w_train)
        w_tilde_val = np.squeeze((1 - alpha_val) * w_tilde_val + alpha_val * w_val)

        if np.ndim(y_train) < 2:
            
            alphas.append(alpha_train)
            weights.append(w_train)
            weights_tilde.append(w_tilde_train)

            output["w_tilde_c1"].append(np.linalg.norm(w_tilde_train[y_train == 1]))
            output["w_tilde_c0"].append(np.linalg.norm(w_tilde_train[y_train == 0]))
            output["w_delta_c1"].append(np.linalg.norm(w_train[y_train == 1]))
            output["w_delta_c0"].append(np.linalg.norm(w_train[y_train == 0]))

        output["epochs"].append(int(epoch + epochs_per_update))

    return model, np.transpose(alphas), np.transpose(weights), np.transpose(weights_tilde), output


def train_val_test_split(X, y, validation_size, test_size, seed=42):

    train_idx, test_idx = train_test_split(np.arange(y.shape[0]), test_size=test_size, random_state=seed, stratify=y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_idx, val_idx = train_test_split(np.arange(y_train.shape[0]), test_size=validation_size, random_state=seed, stratify=y_train)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    return X_train, X_test, X_val, y_train, y_test, y_val


def train_val_test_split_multiclass(X, y, validation_size, test_size, seed=42):

    # Stratify by minority label 
    train_idx, test_idx = train_test_split(np.arange(y.shape[0]), test_size=test_size, random_state=seed, 
                                           stratify=np.argmax(y, axis=1))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_idx, val_idx = train_test_split(np.arange(y_train.shape[0]), test_size=validation_size, random_state=seed, 
                                        stratify=np.argmax(y_train, axis=1))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    return X_train, X_test, X_val, y_train, y_test, y_val


def train_predict_synthetic(X, y, model_fn, loss_fn, smooth_weights, sample_weights_fn, max_epochs, learning_rate,
                            seed=42, verbose=0, pre_train=0, lr_decay=False, adaptive_weights=False):

    if np.ndim(y) > 1:
        X_train, X_test, X_val, y_train, y_test, y_val = train_val_test_split_multiclass(X, y.astype(int), validation_size=0.2, test_size=0.5, seed=seed)
    
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = train_val_test_split(X, y.astype(int), validation_size=0.2, test_size=0.5, seed=seed)
    
    X_train_std, X_test_std, X_val_std = scale_train_test_val(X_train, X_val=X_val, X_test=X_test)
    
    model, betas, weights, weights_tilde, results = _train_synthetic(
        X_train_std, X_val_std, y_train, y_val, model_fn, loss_fn,
        pre_train=pre_train, adaptive_weights=adaptive_weights,
        smooth_weights=smooth_weights, lr_decay=lr_decay,
        learning_rate=learning_rate, 
        verbose=verbose, sample_weights_fn=sample_weights_fn,
        max_epochs=max_epochs, seed=seed
    )

    if np.ndim(y) > 1:
        train_results = multiclass_prediction_results(X_train_std, y_train, model, results, name="train")
        test_results = multiclass_prediction_results(X_test_std, y_test, model, results, name="test")
        val_results = multiclass_prediction_results(X_val_std, y_val, model, results, name="val")

    else:

        train_results = prediction_results(X_train_std, y_train, model, results, name="train")
        test_results = prediction_results(X_test_std, y_test, model, results, name="test")
        val_results = prediction_results(X_val_std, y_val, model, results, name="val")

    output = {
        "results": results,
        "weights_tilde": weights_tilde,
        "weights": weights,
        "X_train": X_train,
        "X_test": X_test,
        "betas": betas
    }

    return model, output 
