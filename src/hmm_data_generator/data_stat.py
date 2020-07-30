from collections import Counter

import numpy as np 
import matplotlib.pyplot as plt


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

	counts = np.zeros((4, 4), dtype=float)
	for x in X:

		x = x[x != 0]

		d = dict(Counter(zip(x[:-1], x[1:])))

		for (a, b), count in d.items():
			counts[int(a) - 1, int(b) - 1] += count

	print("TRANSITION RATES:")
	print(counts / np.sum(counts) * 100, "\n")


def dist_per_timepoint(X, p_fig=None, c=["C0", "C1", "C2", "C3"]):

	stats = np.zeros((X.shape[1], 4))
	for j in range(X.shape[1]):

		x = X[:, j]
		v, c = np.unique(x[x != 0], return_counts=True)

		for n, m in zip(v, c):
			stats[j, int(n - 1)] = m


	plt.figure()
	for i in range(stats.shape[1]):
		plt.plot(stats[:, i], label=f"{i+1}")
		plt.axvline(x=np.argmax(stats[:, i]), c=c[i])

	plt.legend()

	if p_fig is not None:
		plt.savefig(p_fig)

	plt.show()


def main():
	# TODO: 
	# * Compare dist per timepoint to expectations (HMM paper)
	# * Make table comparing transition rates in HMM and real data.

	X = np.load("/Users/sela/Desktop/hmm.npy")

	#distribution(X)
	#transition_rates(X)
	#dist_per_timepoint(X)

	# For comparison.
	X = np.load("/Users/sela/Desktop/Xprep.npy")

	#distribution(X)
	#transition_rates(X)
	dist_per_timepoint(X)


if __name__ == "__main__":
	#comparison_data()
	main()

# {(1.0, 1.0): 8191739, (1.0, 2.0): 21955, (2.0, 2.0): 119319, (2.0, 1.0): 21720, (2.0, 3.0): 4008, (3.0, 3.0): 54355, (3.0, 2.0): 3772, (3.0, 1.0): 916, (1.0, 3.0): 748, (3.0, 4.0): 69, (4.0, 4.0): 1934, (4.0, 1.0): 73, (1.0, 4.0): 7, (4.0, 2.0): 3, (4.0, 3.0): 1, (2.0, 4.0): 1}