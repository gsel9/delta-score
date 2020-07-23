from collections import Counter

import numpy as np 


def comparison_data():
	# Make dataset to compare with synth data.
	X = np.load("/Users/sela/Desktop/recsys_paper/data/screening/X.npy")

	# Require at least 5 obs.
	X_new = []
	for x in X:

		if sum(x != 0) < 5:
			continue 

	# Truncate after cancer.
	I, J = np.where(X == 4)
	for i, row in enumerate(I):

		X[row, int(J[i]) + 1:] = 0

	np.save("/Users/sela/Desktop/Xprep.npy", X)


def distribution(X):

	v, c = np.unique(X[X != 0], return_counts=True)

	print("DISTRIBUTION:")
	print(v)
	print(c, c / sum(c), "\n")


def transition_rates(X):

	x = X[X != 0]

	d = dict(Counter(zip(x[:-1], x[1:])))

	counts = np.zeros((4, 4), dtype=float)
	for (a, b), count in d.items():
		counts[int(a) - 1, int(b) - 1] = count

	print("TRANSITION RATES:")
	print(counts / np.sum(counts) * 100, "\n")


def main():

	X = np.load("/Users/sela/Desktop/hmm.npy")

	distribution(X)
	transition_rates(X)

	# For comparison.
	X = np.load("/Users/sela/Desktop/Xprep.npy")

	distribution(X)
	transition_rates(X)


if __name__ == "__main__":
	#comparison_data()
	main()

# {(1.0, 1.0): 8191739, (1.0, 2.0): 21955, (2.0, 2.0): 119319, (2.0, 1.0): 21720, (2.0, 3.0): 4008, (3.0, 3.0): 54355, (3.0, 2.0): 3772, (3.0, 1.0): 916, (1.0, 3.0): 748, (3.0, 4.0): 69, (4.0, 4.0): 1934, (4.0, 1.0): 73, (1.0, 4.0): 7, (4.0, 2.0): 3, (4.0, 3.0): 1, (2.0, 4.0): 1}