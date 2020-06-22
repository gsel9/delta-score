import numpy as np
from summary import data_summary


SEED = 42


def drop_all_normals(Z):

	Z_copy = Z.copy()

	to_keep = []
	for num, z in enumerate(Z):

		if np.array_equal(np.unique(z[z != 0]), [1]):
			continue

		if np.array_equal(np.unique(z), [0, 1]):
			continue

		to_keep.append(num)

	return Z_copy[to_keep]


def train_test_states(Z, time_lag=4):

	Z_copy = Z.copy()

	t_pred = Z_copy.shape[1] - np.argmax(Z_copy[:, ::-1] != 0, axis=1) - 1

	q = np.copy(Z_copy[range(Z_copy.shape[0]), t_pred])

	# Remove observations in or after prediction window
	for i_row in range(Z_copy.shape[0]):
		Z_copy[i_row, max(0, t_pred[i_row] - time_lag):] = 0

	# Find rows that still contain observations
	valid_rows = np.sum(Z_copy, axis=1) > 0

	return Z_copy[valid_rows], q[valid_rows], t_pred[valid_rows]


def drop_all_normals_but_last(Z):

	Z_train, _, _ = train_test_states(Z)

	to_keep = []
	for num, z in enumerate(Z_train):

		if np.array_equal(np.unique(z[z != 0]), [1]):
			continue

		if np.array_equal(np.unique(z), [0, 1]):
			continue

		to_keep.append(num)

	return Z[to_keep]


def main():

	X_train = np.load("/Users/sela/Desktop/recsys_paper/data/screening/train/X_train.npy")
	X_test = np.load("/Users/sela/Desktop/recsys_paper/data/screening/test/X_test.npy")

	data_summary(X_train)
	data_summary(X_test)
	print()

	X_train_filtered = drop_all_normals_but_last(drop_all_normals(X_train))
	X_test_filtered = drop_all_normals_but_last(drop_all_normals(X_test))

	data_summary(X_train_filtered)
	data_summary(X_test_filtered)

	np.save("/Users/sela/Desktop/recsys_paper/data/screening_filtered/train/X_train.npy", X_train_filtered)
	np.save("/Users/sela/Desktop/recsys_paper/data/screening_filtered/test/X_test.npy", X_test_filtered)


if __name__ == "__main__":
	main()