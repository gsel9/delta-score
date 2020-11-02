import numpy as np

import hashlib

from baseline import oracle, ffill
from matplotlib.pyplot import savefig, subplots
from numpy import sum, sqrt

RESULTS_PATH = "results/"
DS_PATH = "data/small/10p/"


class MAP:

    def __init__(self, M_train, theta=2.5, domain_z=np.arange(1, 5), binary_thresh=2):

        self.M_train = M_train
        self.theta = theta
        self.domain_z = domain_z

        self.z_to_binary_mapping = lambda z: np.array(z) > binary_thresh 

        # Initialize prediction probabilities
        self.__proba_z_precomputed = None
        self.__ds_X_hash = None
        self.__ds_t_hash = None

    def __is_match_ds_hash(self, X, t):
        """Check if hash of (X, t) matches stored

        Checks if the stored hexadecimal hash
        matches the hexademical hash of the input 
        (X, t). 

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        match : bool
        True if match
        """
        if self.__ds_X_hash is None or self.__ds_t_hash is None:
            return False

        if (hashlib.sha1(X).hexdigest() == self.__ds_X_hash) and (hashlib.sha1(t).hexdigest() == self.__ds_t_hash):
            return True

        return False

    def __store_ds_hash(self, X, t):
        """Store hash of dataset.

        Stores a hexadecimal hash of the dataset X used
        in predict_proba.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        self
        Model with stored hash
        """
        self.__ds_X_hash = hashlib.sha1(X).hexdigest()
        self.__ds_t_hash = hashlib.sha1(t).hexdigest()

        return self

    def _loglikelihood(self, X):
        """Compute loglikelihood of X having originated from 
        the fitted profiles (U V^T).

        For all x_i in X, compute the log of the estimated
        likelihood that x_i originated from m_j for j = 1, ..., N
        where N = n_samples_train is the number of samples used in
        training the model. 

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        Returns
        -------
        logL : array_like, shape (n_samples, n_samples_train)
        The logs of the estimated likelihoods.
        """

        N_1 = self.M_train.shape[0]
        N_2 = X.shape[0]

        logL = np.ones((N_2, N_1))

        # Logarithm of (2.14) in project report.
        # Sum over difference between each sample x in X and all samples in M (sum(x - M)).
        for i in range(N_2):
            row_nonzero_cols = X[i] != 0
            eta_i = (X[i, row_nonzero_cols])[None, :] - self.M_train[:, row_nonzero_cols]
            logL[i] = np.sum(-self.theta*np.power(eta_i, 2), axis=1)

        return logL

    def predict_proba(self, X, t):
        """Compute class probabilities.

        For all (x_i, t_i) in (X, t), compute the estimated
        probability that row i will at time t be in the state
        z for z in the domain_z of the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        proba_z_normalized
        The probalities
        """

        # If evaluating several scoring methods subsequently,
        # significant computational time can be saved by storing
        # the class probabilities

        if self.__is_match_ds_hash(X, t):
            return self.__proba_z_precomputed

        X, t = X, t

        # Pre-compute p(x \mid M_hat) for each x \in X.
        logL = self._loglikelihood(X)

        # Compute (2.22) in project report.
        proba_z = np.empty((X.shape[0], self.domain_z.shape[0]))
        for i in range(X.shape[0]):
            # Latter term is p(s_t \mid m_t).
            proba_z[i] = np.exp(logL[i]) @ np.exp(-self.theta * (self.M_train[:, t[i], None] - self.domain_z)**2)

        # Scale to [0, 1] (C in (2.22)).
        proba_z_normalized = proba_z / (np.sum(proba_z, axis=1))[:, None]

        # Store probabilities
        self.__proba_z_precomputed = proba_z_normalized
        self.__store_ds_hash(X, t)

        return proba_z_normalized

    def predict_proba_binary(self, X, t):
        """Compute binary probabilities.

        For all (x_i, t_i) in (X, t), compute the estimated
        probability that row i will at time t be True.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        proba_bin
        The probalities.
        """
        # If evaluating several scoring methods subsequently,
        #  significant computational time can be saved by storing
        #  the class probabilities
        proba_z = self.predict_proba(X, t)

        values_of_z_where_true = [self.z_to_binary_mapping(z) for z in self.domain_z]

        proba_bin = np.sum(proba_z[:, values_of_z_where_true], axis=1).flatten()

        return proba_bin

    def predict(self, X, t, bias_z=None):
        """Predict the most probable state z at time t_i for each (x_i, t_i) in (X, t).

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        bias : array_like, shape (n_states_z, )
        The bias of the model.

        Returns
        -------
        z_states : (n_samples, )
        The predicted states.
        """
        proba_z = self.predict_proba(X, t)

        if bias_z is None:
            return self.domain_z[np.argmax(proba_z, axis=1)]
        
        return self.domain_z[np.argmax(proba_z * bias_z, axis=1)]

    def predict_binary(self, X, t, bias_bin=None):
        """Predict future binary outcome.

        For all (x_i, t_i) in (X, t), predict the most probable
        binary outcome at time t.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        bias : array_like, shape (n_states_e, )
        The bias of the model.

        Returns
        -------
        bin_states : (n_samples, )
        The predicted states.
        """

        proba_bin = self.predict_proba_binary(X, t)

        if bias_bin is None:
            return np.ones_like(proba_bin) * (proba_bin >= 0.5)
        
        return np.ones_like(proba_bin) * (proba_bin >= 1 - bias_bin)


def rmse(y_hat, y):
    mse = sum((y_hat - y) ** 2)
    rmse = sqrt(mse / (y_hat.shape[0]*y_hat.shape[1]))
    return rmse


if __name__ == "__main__":
    # Demo run

    from sklearn.metrics import matthews_corrcoef

    RMSE = []
    MCC = []

    for iter in range(0, 17000, 200):
        # Reconstructed data matrix.
        M_hat = np.load(RESULTS_PATH+f"Y_hat_10p_{iter}_SGMC.npy")
        M_rec = np.load(DS_PATH+"M_rec_10p.npy")

        # M_hat = np.load(RESULTS_PATH+"Y_hat_10p_11400_SGMC_col.npy")
        # M_hat = ffill(np.load(RESULTS_PATH+"Y_rec_10p.npy"))

        # A subset of the sparse data matrix that you wish to predict on (not part of the
        # previously reconstructed data resulting in M_hat).
        Y_test = np.load(DS_PATH+"Y_test_10p.npy")

        # Calculate RMSE
        rmse_ = rmse(M_hat, M_rec)

        # Corresponds to predicting 1 year ahead in time. delta_t = 8/12/... corresponds to
        # predicting 2/3/... years ahead in time.
        delta_t = 4

        # Find time of last observed entry for all rows.
        time_of_prediction = Y_test.shape[1] - np.argmax(Y_test[:, ::-1] != 0, axis=1) - 1

        # The values to be predicted.
        y_true = np.copy(Y_test[range(Y_test.shape[0]), time_of_prediction])

        # Remove observations over time window delta_t.
        for i_row in range(Y_test.shape[0]):
            Y_test[i_row, max(0, time_of_prediction[i_row] - delta_t):] = 0

        # Find rows that still contain observations
        valid_rows = np.sum(Y_test, axis=1) > 0
        y_true = y_true[valid_rows]
        Y_test = Y_test[valid_rows]
        time_of_prediction = time_of_prediction[valid_rows]

        # Predict the states (integers 1=normal-4=cancer).
        estimator = MAP(M_hat, theta=2.5)
        y_pred = estimator.predict(Y_test, time_of_prediction)

        mcc_ = matthews_corrcoef(y_true, y_pred)
        # Performance score.
        print(f"Iter {iter}: MCC={mcc_}, RMSE={rmse_}")

        # # Confusion matrix.
        # cm = np.zeros((4, 4), dtype=int)
        # for i, y in enumerate(y_true):
        #     cm[int(y - 1), int(y_pred[i] - 1)] += 1
        #
        # print(cm)
        RMSE.append(rmse_)
        MCC.append(mcc_)

    _, ax = subplots(1, 2, figsize=(10, 10))
    ax[0].plot(RMSE)
    ax[1].plot(MCC)
    savefig('plots/rmse_vs_mcc.png', dpi=600)
