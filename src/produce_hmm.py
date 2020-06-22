import numpy as np

from sklearn.model_selection import train_test_split

from mask import simulate_mask
from summary import data_summary

from dgd_data_generator import simulate_float_from_named_basis, simulate_dgd


SEED = 42

CONFIGS = {
	"40p": 2.35,
	"30p": 1.85,
	"20p": 1.35,
	"10p": 0.8,
	"5p": 0.48,
	"3p": 0.32
}


def load_sample_pool(subset_size):

	M = np.load("/Users/sela/Desktop/recsys_paper/data/hmm/base_set_300K.npy")

	# NOTE: Select subset for speed-up.
	np.random.seed(SEED)
	idx = np.random.choice(range(M.shape[0]), size=subset_size, replace=False)

	return M[idx]


def main():

	M = load_sample_pool(subset_size=20000)

	# To truncate profiles at empirical dropout times. The file contains a probbaility 
	# vector where each entry of the vector holds a  probability that the female will never be 
	# screened again.
	path_to_dropout = '/Users/sela/phd/data/real/Pdropout_2Krandom.npy'

	num_samples = 10000

	for density in ["40p", "30p", "20p", "10p", "5p", "3p"]:

		# Location where data matrix and masks will be stored.
		path_to_data = f"/Users/sela/Desktop/recsys_paper/data/hmm/{density}"

		mask = simulate_mask(
			M,
			screening_proba=np.array([0.05, 0.15, 0.40, 0.60, 0.20]),
			memory_length=10,
			level=CONFIGS[density],
			random_state=SEED,
			path_dropout=path_to_dropout
		)

		X = mask * M

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

		data_summary(X_train)
		print()

		X_val = X[val_idx]
		M_val = M[val_idx]

		data_summary(X_val)
		print()

		np.save(f'{path_to_data}/train/M_train.npy', M_train)
		np.save(f'{path_to_data}/train/X_train.npy', X_train)

		np.save(f'{path_to_data}/val/M_val.npy', M_val)
		np.save(f'{path_to_data}/val/X_val.npy', X_val)
		
		X_test = X[test_idx]
		M_test = M[test_idx]

		data_summary(X_test)

		np.save(f'{path_to_data}/test/X_test.npy', X_test)
		np.save(f'{path_to_data}/test/M_test.npy', M_test)


def hmm_sample_demo():
	"""
	How to produce synthetic HMM data. 
	"""

	from tqdm import tqdm
	from hmm_data_generator.hmm_generator import simulate_profile

	# HACK: Should also to specify n_timepoints in src/utils.py.
	n_timepoints = 321

	# Number of screening histories/females/samples.
	n_samples = 5

	np.random.seed(SEED)

	D = []
	for num in range(n_samples):

		# Simulate a synth screening profile.
		d = simulate_profile(n_timepoints, init_age=0, age_max=321)

		if sum(d) == 0:
		    continue

		D.append(d)

	#np.save(f"{PATH_TO_DATA}/base_set_300K.npy", np.array(D))


if __name__ == "__main__":
	#main()
	hmm_sample_demo()
