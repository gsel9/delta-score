import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc
from scipy.stats import wasserstein_distance

from . import plot_utils 

plot_utils.setup()


def plot_probability_surface(filename, X, y_true, model, resolution=100):
    
    # define bounds of the domain
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    # define the x and y scale
    x1grid = np.linspace(min1, max1, resolution)
    x2grid = np.linspace(min2, max2, resolution)

    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    X_grid = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model(X_grid).numpy()

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # reshape the predictions back into a grid
    zz = y_pred.reshape(xx.shape)

    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.9, subplots=(1, 1)))

    # plot the grid of x, y and z values as a surface
    c = axis.contourf(xx, yy, zz, cmap='Paired', alpha=0.7)
    fig.colorbar(c)

    labels = np.unique(y_true)
    for label in labels:
        axis.scatter(X[y_true == label, 0], X[y_true == label, 1], cmap='Paired', alpha=0.7, label=f"C{int(label)}")

    plot_utils.format_axis(axis, fig, xlabel="x1", ylabel="x2", grid=True)

    fig.legend()
    fig.tight_layout()
    fig.savefig(f"{filename}_probability.pdf", transparent=True, bbox_inches="tight") 


def plot_decision_surface(filename, X, y_true, model, resolution=100):

    # define bounds of the domain
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    # define the x and y scale
    x1grid = np.linspace(min1, max1, resolution)
    x2grid = np.linspace(min2, max2, resolution)

    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    X_grid = np.c_[xx.ravel(), yy.ravel()]
    p_hat = model(X_grid).numpy()

    if np.ndim(p_hat):
        y_pred = np.argmax(p_hat, axis=1).astype(int)
    else:
        y_pred = (p_hat > 0.5).astype(int)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # reshape the predictions back into a grid
    zz = y_pred.reshape(xx.shape)

    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=0.9, subplots=(1, 1)))

    # plot the grid of x, y and z values as a surface
    axis.contourf(xx, yy, zz, cmap='Paired', alpha=0.7)

    labels = np.unique(y_true)
    for label in labels:
        axis.scatter(X[y_true == label, 0], X[y_true == label, 1], cmap='Paired', alpha=0.7, label=f"C{int(label)}")

    plot_utils.format_axis(axis, fig, xlabel="x1", ylabel="x2", grid=True)

    fig.legend()
    fig.tight_layout()
    fig.savefig(f"{filename}_decision.pdf", transparent=True, bbox_inches="tight") 
