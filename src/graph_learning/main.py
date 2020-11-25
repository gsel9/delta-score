from time import time 

import numpy as np 

from utils import intial_adjacency
from annealing import estimate_adjacency


def interpolation_region_mask(Y):
    
    O = np.zeros_like(Y)
    for i, y in enumerate(Y):
        
        t_start = np.argmax(Y[i] != 0)
        t_end = np.argmax(np.cumsum(Y[i])) + 1
        
        O[i, t_start:t_end] = 1
        
    return O


def synthetic_data_gen():
    
    #M = np.load("../../data/graph_learning/M.npy")
    #Y = np.load("../../data/graph_learning/Y.npy")
    #Z = np.load("../../data/graph_learning/Z.npy")
    #A = np.load("../../data/graph_learning/A.npy")
        
    M = np.load("../../data/M_train.npy")
    Y = np.load("../../data/X_train.npy")
    #Z = np.load("../data/graph_learning/Z.npy")
    #A = np.load("../data/graph_learning/A.npy")
    
    O = interpolation_region_mask(Y)

    return M * O
    
    
def screening_data_gen():
    
    M = np.load("/Users/sela/Desktop/recsys_paper/results/screening/mf/train/train_Xrec.npy")
    Y = np.load("/Users/sela/Desktop/recsys_paper/data/screening/train/X_train.npy")
    O = interpolation_region_mask(Y)

    return M * O


def main():

	# Choose between synthetic and screening data for demonstration run.
	M = synthetic_data_gen()
	#M = screening_data_gen()

	M = M[:20]

	# Initialize random graph.
	A_init = intial_adjacency(n=M.shape[0], p=0.1, seed=42)

	# Increase n_active to increase density in A. 
	# Increase eps to decrease density in A. 
	t0 = time()
	A_hat, M_hat, E, L = estimate_adjacency(A_init, M, method="cf", num_epochs=100, n=5, k=5, alpha=0.3)
	print(E)
	print(L)
	#np.save("A_hat.npy", A_hat)
	print("Duration:", time() - t0)
	print(A_hat)

	np.array_equal(A_init, A_hat), np.sum(A_init), np.sum(A_hat)


if __name__ == "__main__":
	main()