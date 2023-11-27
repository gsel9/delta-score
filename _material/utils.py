import numpy as np 
import pandas as pd 
import tensorflow as tf 

from sklearn.metrics import auc
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_BASE = ""

METADATA = [
	"time_to_target", "target"
]
METADATA_LONGI = [
	"time_to_target", "target", "dt0_density", "dt0_irregularity", 
	"dt1_density", "dt1_irregularity", "dt2_density", "dt2_irregularity"
]

np.random.seed(42)
tf.random.set_seed(42)


def update_alpha(update_step):
	return 1 / (update_step + 1)


class AlphaSchedule:

	def __init__(self, sample_size, thresh=None):

		self.thresh = thresh
		self.counted_correct = np.zeros(sample_size)

	def _beta(self, p_target):

		self.counted_correct[p_target > self.thresh] += 1
		self.counted_correct[p_target <= self.thresh] -= 1
		# Lower bound is zero
		self.counted_correct[self.counted_correct < 0] = 0 

		return 1 / (1 + self.counted_correct) 

	def estimate(self, p_hat, y_true):

		if np.ndim(y_true) > 1:

			p_target = np.sum(y_true * p_hat, axis=1)
			
			if self.thresh is None:
				self.thresh = 1 / y_true.shape[1] 
		else:
			p_target = y_true * p_hat + (1 - y_true) * (1 - p_hat)

			if self.thresh is None:
				self.thresh = 1 / 2 

		return self._beta(p_target)


def _balanced_flip_y(y, flip_y, seed):

	labels, counts = np.unique(y, return_counts=True)
	if labels.size > 2:
		raise ValueError("Not implemented for multiclass")

	generator = check_random_state(seed)

	# Flip a fraction of the minority class 
	n_to_flip = int(round(min(counts) * flip_y))

	y_flipped = np.copy(y)
	for label in labels:

		idx_to_flip = generator.choice(np.squeeze(np.where(y == label)), size=n_to_flip, replace=False)
		y_flipped[idx_to_flip] = 1 - label	

	return y_flipped


def _flip_y(y, flip_y, seed):

	labels = np.unique(y)
	if labels.size > 2:
		raise ValueError("Not intended for multiclass")

	generator = check_random_state(seed)
	flip_mask = generator.uniform(size=y.shape[0]) < flip_y

	y_flipped = np.copy(y)
	y_flipped[flip_mask] = generator.choice(labels, size=flip_mask.sum())

	return y_flipped


def synthetic_data(overlap=0, imbalance=0, noise=0, as_tensor=False, seed=42, keep_balance=False, 
				n_classes=2, n_samples=3000, n_features=2, n_redundant=0):

	if as_tensor:
		raise ValueError("Synth data not longi!")

	if keep_balance:
		X, y = make_classification(n_samples=n_samples, n_features=n_features, flip_y=0, n_redundant=n_redundant,
								class_sep=1 - overlap, weights=[1 - imbalance, imbalance], random_state=seed,
								n_clusters_per_class=1, n_informative=n_features - n_redundant,
								n_classes=n_classes)

		if noise > 0:
			y = _balanced_flip_y(y, noise, seed)

	else:
		X, y = make_classification(n_samples=n_samples, n_features=n_features, flip_y=0, n_redundant=n_redundant,
								class_sep=1 - overlap, weights=[1 - imbalance, imbalance], random_state=seed,
								n_clusters_per_class=1, n_informative=n_features - n_redundant,
								n_classes=n_classes)

		if noise > 0:
			y = _flip_y(y, noise, seed)

	return X.astype(np.float32), y.astype(np.float32)


def delta_auc_score(y_true=None, p_hat=None, delta=None, nbins=50, target=1):

	thresholds = np.linspace(-1, 0, nbins)

	if delta is None:
		delta = delta_score(y_true, p_hat) 

	if target is not None:
		delta = delta[y_true == target]

	correct_clfs = np.ones(thresholds.size) * np.nan 

	for i, tau in enumerate(thresholds):
		correct_clfs[i] = sum(delta <= tau) / delta.shape[0]

	return auc(thresholds, correct_clfs)


def delta_score(y_true, p_pred):

	p_pred = np.squeeze(p_pred)
	y_true = np.squeeze(y_true)
	
	return ((1 - y_true) * (2 * p_pred - 1)) + (y_true * (1 - 2 * p_pred)) 


def validation_split(X, y, validation_size, random_state=42, scale=True):

	train_idx, val_idx = train_test_split(np.arange(y.size),
									      test_size=validation_size, random_state=random_state, stratify=y)

	X_train, X_val = X[train_idx], X[val_idx]
	y_train, y_val = y[train_idx], y[val_idx]

	return X_train, X_val, y_train, y_val, train_idx, val_idx


def filter_all_normal(data, target=None, hpv=False):

	data_c = data.where(data.loc[:, "target"] == target).dropna(how="all")

	target_features = ['dt0_hist_lowgrade', 'dt0_cyt_lowgrade']
	for target_feature in target_features:
		data_c = data_c.where(data_c.loc[:, target_feature] == 0).dropna(how="all")

	if not hpv:
		return data_c

	return filter_hpv(data_c, target=None)


def filter_hpv(data, target=None):

	if target is not None:
		data = data.where(data.loc[:, "target"] == target).dropna(how="all")

	hpv_results = data.loc[:, ['dt0_hpv_pos', 'dt0_hpv_neg']].sum(axis=1)
	
	return data.iloc[hpv_results.values > 0]


def prediction_results(X, y, model, results, name="", delta_thresh=0):

	results[f"p_hat_{name}"].extend(np.squeeze(model(X).numpy()))
	results[f"y_true_{name}"].extend(np.squeeze(y))
	results[f"y_hat_{name}"].extend((np.array(results[f"p_hat_{name}"]) > 0.5).astype(int))
	results[f"delta_{name}"].extend(delta_score(y, np.squeeze(model(X).numpy())))
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


def screening_data(data_dir, fname, as_tensor=False):

	data_train = pd.read_csv(f"{DATA_BASE}/datasets/{data_dir}/{fname}.csv", index_col=0)	
	print(f"* Loaded {data_train.shape[0]} samples")

	y = data_train.loc[:, "target"].values

	if as_tensor:

		X = data_train.loc[:, sorted(list(set(data_train.columns) - set(METADATA_LONGI)))[::-1]].values
		return X.reshape(X.shape[0], 3, 10), y, data_train.loc[:, METADATA_LONGI]

	X = data_train.loc[:, sorted(list(set(data_train.columns) - set(METADATA)))[::-1]].values
	return X, y, data_train.loc[:, METADATA]


def sample_generator(X, y, seeds):

	for idx in idxs:
		yield X[idx], y[idx]


def fbeta_score(tnr, tpr, y_true, beta=2): 

	n1 = np.sum(y_true)
	n0 = np.sum(y_true == 0)

	tp = tpr * n1 
	fn = (1 - tpr) * n1 
	fp = (1 - tnr) * n0 
	beta2 = beta ** 2 
	return (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp) 


class ConvergenceMonitor:

	def __init__(self, delay=5, patience=5, min_delta=0, early_stopping=True):

		self.delay = delay
		self.patience = patience 
		self.min_delta = min_delta
		self.early_stopping = early_stopping

		self.iter = 0 
		self.wait = 0

		self.best = float(np.inf)
		self.current = float(np.inf)
		self.best_weights = None 

	def _is_improvement(self, monitor_value, reference_value):
		return np.less(monitor_value + self.min_delta, reference_value)

	def should_stop(self, current, model):

		self.iter += 1  

		if self.iter < self.delay:
			self.best = current
			return False 
		
		if self.best_weights is None:
			self.best_weights = model.get_weights()

		self.wait += 1

		if self._is_improvement(current, self.best):

			self.best = current
			self.best_weights = model.get_weights()

			self.wait = 0 

		if self.wait >= self.patience:

			if not self.early_stopping:
				return False

			model.stop_training = True
			model.set_weights(self.best_weights)

			return True 

		return False 


class DecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

	def __init__(self, init_learning_rate, pre_train=0):
		self.learning_rate = init_learning_rate
		self.pre_train = pre_train 

		self.iter = 0 

	def __call__(self, step):
		
		self.iter += 1 
		
		if self.iter <= 5:
			return 0.1 
			
		if 5 < self.iter < 10:
			return 0.01 

		return 1e-3