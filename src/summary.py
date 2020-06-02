import numpy as np 


def profile_density(Z):

    densities = 0
    for num, row in enumerate(Z):
    
        z_tr = row[:np.argmax(np.cumsum(row))]

        densities += np.count_nonzero(z_tr) / z_tr.size

    return densities / (num + 1)


def data_summary(X):

    print("Shape:", np.shape(X))
    print("Profile density:", profile_density(X))
    
    vals, cnts = np.unique(X[X != 0], return_counts=True)
    print("Domain:", vals)
    print("Domtain count:", cnts)
