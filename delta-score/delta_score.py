import numpy as np 


# TODO: input checks 
def delta_score(y_true, p_pred):
	"""

	Args:
		y_true (array-like (n_samples, n_columns)): 
		p_pred (array-like (n_samples, n-columns)): 

	Returns:
		(array-like) the delta score value.

	"""

	y_true = y_true.astype(int)
	p_pred = p_pred.astype(float)

	y_true = y_true.squeeze()
	p_pred = p_pred.squeeze()

	p_true = np.sum(y_true * p_pred, axis=1)
	p_max_false = np.max((1 - y_true) * p_pred, axis=1)

	return p_max_false - p_true
