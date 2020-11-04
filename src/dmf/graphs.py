import numpy as np 
from scipy.sparse import csgraph


def row_graph(D, k):

	NN = np.argsort(D, axis=1)
	kNN = NN[:, 1:(k + 1)]

	N, k = np.shape(kNN)

	G = np.zeros((N, N))

	for i in range(N):
		for l in range(k):

			G[i, kNN[i, l]] = 1
			G[kNN[i, l], i] = 1

	np.fill_diagonal(G, 0)

	return G * np.exp(-1.0 * D)


def column_graph(num_nodes, weights=tuple([1])):

	A = np.zeros((num_nodes, num_nodes))

	for l in range(1, len(weights) + 1):

		o = weights[l - 1] * np.ones(num_nodes - l)
		A += np.diag(o, l) + np.diag(o, -l)

	np.fill_diagonal(A, 0)

	return A


if __name__ == "__main__":
	from sklearn.metrics.pairwise import cosine_distances

	DS_PATH = "data/small/10p/"

	# Distance matrix from sparse data matrix.
	D = cosine_distances(np.load("Y_rec.npy"))

	# Adjacency matrices.
	Ar = row_graph(D, 10)
	Ac = column_graph(340, weights=[i for i in np.exp(-np.arange(5))])

	# Laplacians.
	Lr = csgraph.laplacian(Ar, normed=True)
