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
    
    M = np.load("../data/M_train.npy")
    Y = np.load("../data/X_train.npy")
    
    O = interpolation_region_mask(Y)

    return M * O
    
    
def screening_data_gen():
    
    M = np.load("/Users/sela/Desktop/recsys_paper/results/screening/mf/train/train_Xrec.npy")
    Y = np.load("/Users/sela/Desktop/recsys_paper/data/screening/train/X_train.npy")
    O = interpolation_region_mask(Y)

    return M * O


def main():

	# Choose between synthetic and screening data for demonstration run.
	#M = synthetic_data_gen()
	M = screening_data_gen()

	n = 6
	A_init = intial_adjacency(n=n, p=0.4, seed=42)
	print(A_init)
	print()

	# Increase n_active to increase density in A. 
	# Increase eps to decrease density in A. 
	A_hat = estimate_adjacency(A_init, M[:n], num_epochs=10, N=3, eps=0.5)
	#np.save("A_hat.npy", A_hat)
	print(A_hat)

	np.array_equal(A_init, A_hat), np.sum(A_init), np.sum(A_hat)


if __name__ == "__main__":
	main()