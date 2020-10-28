from typing import Tuple, List
import numpy as np


from sklearn.model_selection import train_test_split 


def train_val_test_splitting(X, val_size=0.1, test_size=0.1, M=None, stratify=None, seed=42,
                             path_to_files=None):

    train_idx, val_idx = train_test_split(np.arange(X.shape[0]),
                                          test_size=int(X.shape[0] * val_size),
                                          random_state=seed,
                                          stratify=None) 

    train_idx, test_idx = train_test_split(train_idx,
                                           test_size=int(X.shape[0] * test_size),
                                           random_state=seed,
                                           stratify=None) 

    X_train = X[train_idx].astype(float)
    X_test = X[test_idx].astype(float)
    X_val = X[val_idx].astype(float)

    if path_to_files is None:
        return X_train, X_test, X_val, M[train_idx].astype(float), M[test_idx].astype(float), M[val_idx].astype(float)

    np.save(f"{path_to_files}/Y_rec.npy", X_train)
    np.save(f"{path_to_files}/Y_val.npy", X_test)
    np.save(f"{path_to_files}/Y_test.npy", X_val)

    if M is not None:
        np.save(f"{path_to_files}/M_rec.npy", M[train_idx].astype(float))
        np.save(f"{path_to_files}/M_val.npy", M[test_idx].astype(float))
        np.save(f"{path_to_files}/M_test.npy", M[val_idx].astype(float))


def train_val_test_subsample(X_train, X_test, X_val, 
                             N, val_size, test_size, train_size,
                             M_train=None, M_test=None, M_val=None, 
                             p=None, path_to_files=None, seed=42):

    np.random.seed(seed)

    train_idx = np.random.choice(range(X_train.shape[0]), 
                                 size=int(N * train_size),
                                 replace=False, p=p)

    test_idx = np.random.choice(range(X_test.shape[0]), 
                                 size=int(N * test_size),
                                 replace=False, p=p)

    val_idx = np.random.choice(range(X_val.shape[0]), 
                                 size=int(N * val_size),
                                 replace=False, p=p)

    ss_X_train = X_train[train_idx]
    ss_X_test = X_test[test_idx]
    ss_X_val = X_val[val_idx]

    np.save(f"{path_to_files}/Y_rec.npy", ss_X_train)
    np.save(f"{path_to_files}/Y_val.npy", ss_X_test)
    np.save(f"{path_to_files}/Y_test.npy", ss_X_val)

    print("Rec density:", np.mean(ss_X_train != 0))

    if M_train is not None:
        np.save(f"{path_to_files}/M_rec.npy", M_train[train_idx].astype(float))

    if M_test is not None:
        np.save(f"{path_to_files}/M_val.npy", M_test[test_idx].astype(float))
        
    if M_val is not None:
        np.save(f"{path_to_files}/M_test.npy", M_val[val_idx].astype(float))


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
