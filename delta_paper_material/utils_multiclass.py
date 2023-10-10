import numpy as np 
import pandas as pd 
import tensorflow as tf 

from sklearn.metrics import auc
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_BASE = ""


def _balanced_flip_multiclass_y(y, flip_y, seed):

	labels, counts = np.unique(y, return_counts=True)
	
	generator = check_random_state(seed)

	# Flip a fraction of the minority class 
	n_to_flip = int(round(min(counts) * flip_y))

	# Make divisible by number of alternative labels 
	n_to_flip = n_to_flip - (n_to_flip % (len(labels) - 1))

	y_flipped = np.copy(y)
	for label in labels:

		idx_to_flip = generator.choice(np.squeeze(np.where(y == label)), size=n_to_flip, replace=False)

		# Replace target label by an equal number of labels from other classes 
		alternatives = labels[labels != label]

		splits = np.array_split(idx_to_flip, len(alternatives))
		for i, split in enumerate(splits):
		
			y_flipped[split] = alternatives[i]

	return y_flipped


def _flip_y(y, flip_y, seed):

	generator = check_random_state(seed)
	flip_mask = generator.uniform(size=y.shape[0]) < flip_y

	y_flipped = np.copy(y)
	y_flipped[flip_mask] = generator.choice(np.unique(y), size=flip_mask.sum())

	return y_flipped


def synthetic_data_multiclass(overlap=0, imbalance=0.05, noise=0, as_tensor=False, seed=42, n_samples=3000, 
							  n_features=2, n_redundant=0, ohe_y=True, keep_balance=True):
							  
	if keep_balance:

		X, y = make_classification(n_samples=n_samples, n_features=n_features, flip_y=0, n_redundant=n_redundant,
							  	   class_sep=1 - overlap, n_informative=n_features - n_redundant,
								   weights=[0.5 * (1 - imbalance), 0.5 * (1 - imbalance), imbalance], 
								   random_state=seed, n_classes=3, n_clusters_per_class=1)

		if noise > 0:
			y = _balanced_flip_multiclass_y(y, noise, seed)

	else:

		X, y = make_classification(n_samples=n_samples, n_features=n_features, flip_y=0, n_redundant=n_redundant,
							  	   class_sep=1 - overlap, n_informative=n_features - n_redundant,
								   weights=[0.5 * (1 - imbalance), 0.5 * (1 - imbalance), imbalance], 
							   	   random_state=seed, n_classes=3, n_clusters_per_class=1)
		
		if noise > 0:
			y = _flip_y(y, noise, seed)

	if ohe_y:

		ohe = OneHotEncoder(sparse=False, dtype=np.float32)
		return X.astype(np.float32), ohe.fit_transform(y[:, None])

	return X.astype(np.float32), y.astype(np.float32)


def delta_score_multiclass(y_true, p_pred):

	y_true = y_true.astype(int)
	p_pred = p_pred.astype(float)

	p_true = np.sum(y_true * p_pred, axis=1)
	p_max_false = np.max((1 - y_true) * p_pred, axis=1)

	return p_max_false - p_true


def delta_auc_score_multiclass(y_true=None, p_hat=None, delta=None, nbins=50, target=1):

	thresholds = np.linspace(-1, 0, nbins)

	if delta is None:
		delta = delta_score_multiclass(y_true, p_hat) 

	if target is not None:
		delta = delta[y_true == target]

	correct_clfs = np.ones(thresholds.size) * np.nan 

	for i, tau in enumerate(thresholds):
		correct_clfs[i] = sum(delta <= tau) / delta.shape[0]

	return auc(thresholds, correct_clfs)


def multiclass_prediction_results(X, y, model, results, name="", delta_thresh=0):

	_p_hat = model(X).numpy()

	results[f"p_hat_{name}"].extend(np.sum(y * _p_hat, axis=1))
	results[f"p_hat_minor_{name}"].extend(_p_hat[:, -1])
	results[f"y_true_{name}"].extend(np.argmax(y, axis=1))
	results[f"y_hat_{name}"].extend(np.argmax(_p_hat, axis=1))
	results[f"delta_{name}"].extend(delta_score_multiclass(y, _p_hat))
	results[f"is_correct_{name}"].extend((np.array(results[f"delta_{name}"]) < delta_thresh).astype(int)) 

	for key, values in results.items():
		results[key] = np.array(values)

	return results


def model_results(results_dir, results_fname, data_dir=None, data_fname=None):

	results = pd.read_csv(f"{results_dir}/{results_fname}.csv", index_col=0)

	if data_dir is not None:

		features = pd.read_csv(f"{DATA_BASE}/datasets/{data_dir}/{data_fname}.csv", index_col=0)
		
		return results.join(features.loc[results.index], rsuffix="_feat") 

	return results
