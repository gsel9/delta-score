import numpy as np 

from sklearn.metrics import auc 


# TODO: input checks 
def sample_coverage(delta_scores, n_thresholds):

	threshold = np.linspace(-1, 1, n_thresholds)
	coverage = np.ones(n_thresholds) * float(np.nan)

	for i, tau in enumerate(threshold):
		coverage[i] = sum(delta_scores < tau) / delta_scores.shape[0]

	return threshold, coverage


# TODO: input checks 
def delta_auc_score(y_true, p_pred, n_thresholds):

	threshold, coverage = sample_coverage(delta_score(y_true, p_pred), n_thresholds)

	return auc(threshold, coverage)