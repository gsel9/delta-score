from typing import Tuple, List
import numpy as np


def train_val_split(X: np.ndarray, forecast: int = 4) -> Tuple:
        """Divide sapmles into training and validation sets.
        Args:
        Returns:
        """
        
        O_val = np.zeros_like(X)
        O_train = np.zeros_like(X)

        O_val[X.nonzero()] = 1
        O_train[X.nonzero()] = 1

        # Time of last observation.
        t_censoring = X.shape[1] - np.argmax(X[:, ::-1] != 0, axis=1) - 1

        for i, t_max in enumerate(t_censoring):

            # Start prediction window.
            t_init_val = max(0, t_max - forecast) + 1

            O_train[i, t_init_val:] = 0
            O_val[i, :t_init_val] = 0

        valid_rows = np.logical_and(np.sum(O_train > 0, axis=1), np.sum(O_val > 0, axis=1))

        return O_train, O_val, valid_rows


def train_val_split_forecasting(X: np.ndarray, forecast: List[int] = [4, 8, 12]) -> Tuple:
        """Divide sapmles into training and validation sets.

        Args:

        Returns:
        """
        
        O_val_1yf = np.zeros_like(X)
        O_val_2yf = np.zeros_like(X)
        O_val_3yf = np.zeros_like(X)

        O_val_1yf[X.nonzero()] = 1
        O_val_2yf[X.nonzero()] = 1
        O_val_3yf[X.nonzero()] = 1

        O_train = np.zeros_like(X)
        O_train[X.nonzero()] = 1

        # Time of last observation.
        t_censoring = X.shape[1] - np.argmax(X[:, ::-1] != 0, axis=1) - 1
   
        for i, t_max in enumerate(t_censoring):

            # Start prediction window.
            t_init_val = max(0, t_max - forecast[-1]) + 1
           
            # Leave space for largest prediction window.
            O_train[i, t_init_val:] = 0

            # Kill signal from training scores.
            O_val_1yf[i, :t_init_val] = 0
            O_val_2yf[i, :t_init_val] = 0
            O_val_3yf[i, :t_init_val] = 0

            # Adjust size of prediction windows.
            O_val_1yf[i, t_init_val + forecast[0]:] = 0
            O_val_2yf[i, t_init_val + forecast[1]:] = 0
            O_val_3yf[i, t_init_val + forecast[2]:] = 0

        valid_rows = np.logical_and(np.sum(O_train > 0, axis=1), np.sum(O_val_1yf > 0, axis=1))
            
        return O_train, O_val_1yf, O_val_2yf, O_val_3yf, valid_rows


if __name__ == "__main__":

    np.random.seed(42)
    X = np.random.randint(0, 5, size=(2, 10))
    X[0, -2:] = 0

    print("X:")
    print(X)
    print()
    O_train, O_val_1yf, O_val_2yf, O_val_3yf = train_val_split(X, forecast=[1, 2, 3])

    print("TRAINING")
    print(X * O_train)
    print()

    print("FORECAST 1Y")
    print(X * O_val_1yf)
    print()

    print("FORECAST 2Y")
    print(X * O_val_2yf)
    print()

    print("FORECAST 3Y")
    print(X * O_val_3yf)
