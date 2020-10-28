import numpy as np

from sklearn.model_selection import train_test_split

from mask import simulate_mask
from summary import data_summary

import scipy.stats as st 

from splitting import train_val_test_splitting, train_val_test_subsample

from dgd_data_generator import simulate_float_from_named_basis, simulate_dgd


SEED = 42


def start_end_times(N, T):

	x = np.arange(T)

	p_start = st.exponnorm.pdf(x=x, K=8.76, loc=9.8, scale=7.07)
	t_start = np.random.choice(x, size=N, p=p_start / sum(p_start))

	p_cens = st.exponweib.pdf(x=x, a=513.28, c=4.02, loc=-992.87, scale=707.63)
	t_cens = np.random.choice(x, size=N, p=p_cens / sum(p_cens))

	i = t_start < t_cens 
	t_cens = t_cens[i]
	t_start = t_start[i]

	assert np.all(t_start < t_cens)

	return t_start, t_cens


def censoring(t_start, t_cens, D, missing=0):

	for i, (t_start_i, t_cens_i) in enumerate(zip(t_start, t_cens)):

		D[i, :t_start_i] = missing
		D[i, t_cens_i:] = missing

	return D


def produce_dataset(N, T, r, seed, level, memory_length=10, missing=0):

	t_start, t_cens = start_end_times(N=N, T=T)

	M = simulate_float_from_named_basis(
		basis_name='simple_peaks', 
		N=len(t_start), 
		T=T, 
		K=r, 
		domain=[1, 4], 
		random_state=seed
	)

	D = simulate_dgd(
		M, 
		domain_z=np.arange(1, 5),
		theta=2.5,
		random_state=seed
	)

	O = simulate_mask(
		D,
		screening_proba=np.array([0.05, 0.15, 0.4, 0.6, 0.2]),
		memory_length=memory_length,
		level=level,
		random_state=seed
	)

	Y = D * O
	Y = censoring(t_start, t_cens, Y, missing=missing)

	valid_rows = np.count_nonzero(Y, axis=1) > 1

	return M[valid_rows], Y[valid_rows]


def main():

	sparsity_levels = {
		#"50p": 3,
		#"40p": 2,
		#"30p": 1.5,
		#"20p": 1,
		#"10p": 0.8,
		"5p": 0.5
	}

	for key, level in sparsity_levels.items():
		M, X = produce_dataset(50000, T=340, r=5, seed=521, level=level, memory_length=10)

		X_train, X_test, X_val, M_train, M_test, M_val = train_val_test_splitting(
			X=X, M=M, val_size=0.3, test_size=0.3, stratify=None
		)

		train_val_test_subsample(
			X_train=X_train,
			X_test=X_test,
			X_val=X_val,
			M_train=M_train,
			M_test=M_test,
			M_val=M_val,
			N=10000,
			val_size=0.2, test_size=0.2, train_size=0.8,
			path_to_files="/Users/sela/Desktop",
			p=None,
			seed=42
		)


if __name__ == "__main__":
	#main() 

	Y_train = np.load(f'/Users/sela/Desktop/data_sanketh/Y_rec_10p.npy')
	M_train = np.load(f'/Users/sela/Desktop/data_sanketh/M_rec_10p.npy')

	#print(Y_train.shape, M_train.shape, Y_test.shape, M_test.shape)

	import matplotlib.pyplot as plt

	path_to_data = "/Users/sela/Desktop/data_sanketh"

	plt.figure()
	plt.imshow(Y_train, aspect="auto")
	plt.ylabel("Females")
	plt.xlabel("Time")
	plt.colorbar()
	plt.savefig(f"{path_to_data}/sparse_matrix.pdf")

	plt.figure()
	plt.imshow(M_train, aspect="auto")
	plt.ylabel("Females")
	plt.xlabel("Time")
	plt.colorbar()
	plt.savefig(f"{path_to_data}/original_matrix.pdf")

	_, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
	for i, axis in enumerate(axes.ravel()):

		x = Y_train[i].copy()
		t_start = np.argmax(x != 0) - 1
		t_max = np.argmax(np.cumsum(x)) + 1
		x[x == 0] = np.nan

		m = np.ones_like(x) * np.nan
		m[t_start:t_max] = M_train[i, t_start:t_max].copy()

		axis.plot(x, "o", label="States")
		axis.plot(m, label="Risk")
		axis.set_xlabel("Time")

		axis.legend()

	plt.tight_layout()
	plt.savefig(f"{path_to_data}/samples.pdf")
