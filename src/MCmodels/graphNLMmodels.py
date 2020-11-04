import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class basic_graph:
    def __init__(self, A = None):
        self.A = A

    def compute_laplacian(self):
        self.laplacian_mat = self.degree_mat-self.adjacency_mat

    def compute_diffusion_kernel(self):
        self.diffusion_kernel = 0

class graphNLM(basic_graph):
    def __init__(self, X, mask):
        """
        generate graph from the rows of mat by computing distance between rows
        """
        m = X.shape[0]
        n = X.shape[1]
        X_imputed = self.__impute_rows(X, mask)
        A = squareform(torch.pdist(X_imputed, self.__graph_norm))
        super().__init__(A)

    def __impute_rows(self, X, mask):
        """
        compute norm between
        """
        # TODO: for now we just pass X.*mask,
        #  later we need to implement some imputation method
        return  torch.matmul(X, mask)

    def __graph_norm(self, x1, x2):
        """
        compute norm between two time series on the graph.
        Degault: l2 norm
        """
        return torch.norm(x1,x2)

    def __apply_nlm(self, X):
        X_filtered = torch.matmul(self.diffusion_kernel,X)
        return X_filtered