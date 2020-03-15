from typing import Tuple
import numpy as np


def train_val_split(X,
                    prediction_window,
                    method='last_observed',
                    seed=42) -> Tuple:
        """Divide sapmles into training and validation sets.

        Args:

        Returns:
        """
        
        O_val = np.zeros_like(X)
        O_train = np.zeros_like(X)

        O_val[X.nonzero()] = 1
        O_train[X.nonzero()] = 1

        # Find time of last observed entry for all rows
        if method == 'last_observed':
            time_of_prediction = X.shape[1] - np.argmax(X[:, ::-1] != 0, axis=1) - 1

        # Find time as nearest (in abs. value) nonzero intro to random integer
        elif method == 'random':
            np.random.seed(seed)
            time_of_prediction = np.array([np.argmin(np.abs(np.random.randint(0, X.shape[1]) - np.argwhere(x != 0))) 
                                           for x in X], dtype=np.int)
        else:
            raise ValueError(f"Invalid split method: {method}")

        for i, t_max in enumerate(time_of_prediction):

            val_columns = max(0, t_max - prediction_window)

            O_val[i, val_columns:] = 0
            O_train[i, :val_columns] = 0

        # Retain only rows that still contain observations
        valid_rows = np.sum(O_train, axis=1) > 0

        O_val = O_val[valid_rows, :]
        O_train = O_train[valid_rows, :]

        return O_train, O_val
