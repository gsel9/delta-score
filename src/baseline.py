import numpy as np
import pandas as pd


def ffill(Y_test, time_of_prediction):

	Y_nan = Y_test.copy().astype(float)
	Y_nan[Y_test == 0] = np.nan

	Y_filled = pd.DataFrame(Y_nan).fillna(axis=1, method='ffill')
	Y_filled = np.nan_to_num(Y_filled.values, 0)

	return Y_filled[range(Y_filled.shape[0]), self.time_of_prediction]


def oracle(M_test):

	return np.round(M_test)
