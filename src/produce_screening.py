import numpy as np 

from sklearn.model_selection import train_test_split

from summary import data_summary


SEED = 42


def main():
	
	X = np.load("/Users/sela/Desktop/recsys_paper/data/screening/X.npy")

	train_idx, test_idx = train_test_split(range(X.shape[0]), test_size=0.2, random_state=42)

	X_train = X[train_idx]
	X_test = X[test_idx]

	data_summary(X_train)
	data_summary(X_test)

	np.save("/Users/sela/Desktop/recsys_paper/data/screening/train/X_train.npy", X_train)
	np.save("/Users/sela/Desktop/recsys_paper/data/screening/test/X_test.npy", X_test)


if __name__ == "__main__":
	main()