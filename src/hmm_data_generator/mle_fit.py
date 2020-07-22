"""Fit probability distributions to censoring times and times for initial screening.

https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
"""

import warnings
from tqdm import tqdm

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt


# Create models from data
def best_fit_distribution(data):
	"""Model data by finding best fit distribution to data"""

	x, y = np.unique(data, return_counts=True)

	# Get histogram of original data
	#y, x = np.histogram(data, bins=100, density=True)
	#x = (x + np.roll(x, -1))[:-1] / 2.0

	# Distributions to check
	DISTRIBUTIONS = [
		st.alpha,
		st.anglit,
		st.arcsine,
		st.beta,
		st.betaprime,
		st.bradford,
		st.burr,
		st.cauchy,
		st.chi,
		st.chi2,
		st.cosine,
        st.erlang,
        st.expon,
        st.exponnorm,
        st.exponweib,
        st.exponpow,
        st.f,
        st.fisk,
        st.foldcauchy,
        st.foldnorm,
        st.frechet_r,
        st.frechet_l,
        st.gausshyper,
        st.gamma,
        st.gilbrat,
        st.gompertz,
        st.halfcauchy,
        st.halflogistic,
        st.halfnorm,
        st.halfgennorm,
        st.hypsecant,
        st.invgamma,
        st.invgauss,
        st.invweibull,
        st.johnsonsb,
        st.johnsonsu,
        st.ksone,
        st.kstwobign,
        st.laplace,
        st.levy,
        st.logistic,
        st.loggamma,
        st.loglaplace,
        st.lognorm,
        st.lomax,
        st.maxwell,
        st.norm,
        st.pareto,
        st.powerlognorm,
        st.powernorm,
        st.rayleigh,
        st.rice,
        st.t,
        st.vonmises,
        st.wald,
	  	# NOTE: Should try discrete distributions.
		# st.bernoulli,
		# st.betabinom,
		# st.binom,
		# st.boltzmann,
		# st.dlaplace,
		# st.geom,
		# st.hypergeom,
		# st.logser,
		# st.nbinom,
		# st.planck,
		# st.poisson,
		# st.randint,
		# st.skellam,
		# st.zipf,
		# st.yulesimon
	]

	# Best holders default
	best_distribution = st.norm
	best_params = (0.0, 1.0)
	best_sse = np.inf

	# Estimate distribution parameters from data
	for distribution in tqdm(DISTRIBUTIONS):

		# Ignore warnings from data that can't be fit
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore')

			# fit dist to data
			params = distribution.fit(data)

			# Separate parts of parameters
			arg = params[:-2]
			loc = params[-2]
			scale = params[-1]

			# Calculate fitted PDF and error with fit in distribution
			pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
			sse = sum((y - pdf) ** 2)

			# identify if this distribution is better
			if best_sse > sse > 0:

			    best_distribution = distribution
			    best_params = params
			    best_sse = sse

	return best_distribution.name, best_params


def make_pdf(dist, params, size=10000):
	"""Generate distributions's Probability Distribution Function """

	# Separate parts of parameters
	arg = params[:-2]
	loc = params[-2]
	scale = params[-1]

	# Get sane start and end points of distribution
	#start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
	#end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
	#np.linspace(start, end, size)

	# Build PDF
	x = np.arange(340) 
	y = dist.pdf(x, loc=loc, scale=scale, *arg)
	
	return x, y


def select_censoring_distribution():

	X = np.load("/Users/sela/Desktop/recsys_paper/data/screening/X_orig.npy")
	data = np.argmax(np.cumsum(X, axis=1), axis=1)

	# Find best fit distribution
	best_fit_name, best_fit_params = best_fit_distribution(data)
	best_dist = getattr(st, best_fit_name)

	# Make PDF with best params 
	x, y = make_pdf(best_dist, best_fit_params) 

	param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
	param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
	dist_str = '{}({})'.format(best_fit_name, param_str)
	#print(param_str)
	print("SELECTED:")
	print(dist_str)


def select_init_screening_distribution():

	X = np.load("/Users/sela/Desktop/recsys_paper/data/screening/X_orig.npy")
	data = np.argmax(X != 0, axis=1)

	# Find best fit distribution
	best_fit_name, best_fit_params = best_fit_distribution(data)
	best_dist = getattr(st, best_fit_name)

	# Make PDF with best params 
	x, y = make_pdf(best_dist, best_fit_params)

	param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
	param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
	dist_str = '{}({})'.format(best_fit_name, param_str)
	#print(param_str)
	print("SELECTED:")
	print(dist_str)


def eval_censoring_distribution():

	# plt.figure()
	# X = np.load("/Users/sela/Desktop/recsys_paper/data/screening/X_orig.npy")
	# t_end = np.argmax(np.cumsum(X, axis=1), axis=1)
	# v, c = np.unique(t_end, return_counts=True)
	# plt.bar(v, c / sum(c))

	y = st.exponweib.rvs(a=513.28, c=4.02, loc=-992.87, scale=707.63, size=6000)
	y = y.astype(int)
	print(y)
	# v, c = np.unique(y, return_counts=True)
	# plt.bar(v, c / sum(c), alpha=0.7)
	
	# y = st.exponweib.pdf(x=np.arange(0, 340), 
	# 					 a=513.28, c=4.02, loc=-992.87, scale=707.63)
	# plt.plot(y)
	# plt.show()


def eval_init_screening_distribution():

	plt.figure()
	X = np.load("/Users/sela/Desktop/recsys_paper/data/screening/X_orig.npy")
	t_start = np.argmax(X != 0, axis=1)
	v, c = np.unique(t_start, return_counts=True)
	plt.bar(v, c / sum(c))

	y = st.exponnorm.rvs(K=8.76, loc=9.80, scale=7.07, size=6000)
	y = y.astype(int)
	v, c = np.unique(y, return_counts=True)
	plt.bar(v, c / sum(c), alpha=0.7)
	
	y = st.exponnorm.pdf(x=np.arange(0, 340), K=8.76, loc=9.80, scale=7.07)
	plt.plot(y)
	plt.show()


if __name__ == "__main__":
	#select_init_screening_distribution()
	eval_init_screening_distribution()
	#select_censoring_distribution()
	#eval_censoring_distribution()
