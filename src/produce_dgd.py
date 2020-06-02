import numpy as np

from sklearn.model_selection import train_test_split

from mask import simulate_mask
from summary import data_summary

from dgd_data_generator import simulate_float_from_named_basis, simulate_dgd


SEED = 42

CONFIGS = {
	"40p": 1.76,
	"30p": 1.43,
	"20p": 1.1,
	"10p": 0.67,
	"5p": 0.412,
	"3p": 0.285
}


def main():

	# To truncate profiles at empirical dropout times. The file contains a probbaility 
	# vector where each entry of the vector holds a probability that the female will never be 
	# screened again.
	path_to_dropout = '/Users/sela/phd/data/real/Pdropout_2Krandom.npy'

	num_samples = 10000

	for density in ["40p", "30p", "20p", "10p", "5p", "3p"]:

		# Location where data matrix and masks will be stored.
		path_to_data = f"/Users/sela/Desktop/recsys_paper/data/dgd/{density}"

		M = simulate_float_from_named_basis(
			basis_name='simple_peaks', 
			N=20000, 
			T=321, 
			K=5, 
			domain=[1, 4], 
			random_state=SEED
		)

		D = simulate_dgd(
			M, 
			domain_z=np.arange(1, 5),
			theta=2.5,
			random_state=SEED
		)

		mask = simulate_mask(
			D,
			screening_proba=np.array([0.05, 0.15, 0.40, 0.60, 0.20]),
			memory_length=10,
			level=CONFIGS[density],
			random_state=SEED,
			path_dropout=path_to_dropout
		)

		X = mask * D

		# Keep only samples with at least one score.
		valid_rows = np.sum(X, axis=1) > 0
		X = X[valid_rows]
		M = M[valid_rows]

		train_idx, test_idx = train_test_split(range(num_samples), test_size=0.2, random_state=42)

		cand_test_idx = set(range(X.shape[0])) - set(train_idx) - set(test_idx)

		np.random.seed(42)
		val_idx = np.random.choice(list(cand_test_idx), size=2000, replace=False)
		
		X_train = X[train_idx]
		M_train = M[train_idx]

		#data_summary(X_train)
		#print()

		X_val = X[val_idx]
		M_val = M[val_idx]

		#data_summary(X_val)
		#print()

		#np.save(f'{path_to_data}/train/M_train.npy', M_train)
		#np.save(f'{path_to_data}/train/X_train.npy', X_train)

		# NOTE: Not using validation set since doing k-fold CV.
		# Used training data sub-sample anyway so therefore not combining train with val.
		#np.save(f'{path_to_data}/val/M_val.npy', M_val)
		#np.save(f'{path_to_data}/val/X_val.npy', X_val)

		X_test = X[test_idx]
		M_test = M[test_idx]

		#data_summary(X_test)

		#np.save(f'{path_to_data}/test/X_test.npy', X_test)
		#np.save(f'{path_to_data}/test/M_test.npy', M_test)


if __name__ == "__main__":
	main()
