from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

K_vals = [25, 50, 100]
P_vals = [25, 50, 100]
Ks = list(chain(*[[elm] * len(P_vals) for elm in K_vals]))
Ps = list(chain(*[P_vals for _ in K_vals]))
name_vi = [7.75, 8.05, 7.95, 8.26, 7.75, 8.05, 8.10, 8.05, 7.46]
tv_vi = [5.75, 5.29, 5.08, 5.52, 5.19, 5.39, 5.18, 5.39, 5.36]
name_purity = [74.5, 81.59, 88.25, 76.15, 84.01, 82.97, 89.46, 79.95, 66.46]
tv_purity = [62.5, 76.25, 87.92, 74.17, 78.33, 83.75, 90.42, 70, 57.5]

def best_fit_line(X: np.array, y: np.array) -> np.array:
    n, d = X.shape
    assert len(y.shape) == 1
    X = np.concatenate([X, np.ones((n, 1))], axis=1)
    y = y.reshape(-1, 1)
    coeffecient = np.linalg.inv(X.T@X) @ X.T @ y
    return coeffecient


K_vals = [25, 50, 100]
P_vals = [25, 50, 100]
Ks = list(chain(*[[elm] * len(P_vals) for elm in K_vals]))
Ps = list(chain(*[P_vals for _ in K_vals]))
name_vi = [7.75, 8.05, 7.95, 8.26, 7.75, 8.05, 8.10, 8.05, 7.46]
tv_vi = [5.75, 5.29, 5.08, 5.52, 5.19, 5.39, 5.18, 5.39, 5.36]
name_purity = [74.5, 81.59, 88.25, 76.15, 84.01, 82.97, 89.46, 79.95, 66.46]
tv_purity = [62.5, 76.25, 87.92, 74.17, 78.33, 83.75, 90.42, 70, 57.5]

fig = plt.figure(figsize=(10, 10))
grid_layout = 22


# retrieve = lambda x: [k for k,v in globals().items() if v == x][0]
def best_fit_line(X: np.array, y: np.array) -> np.array:
    n, d = X.shape
    assert len(y.shape) == 1
    X = np.concatenate([X, np.ones((n, 1))], axis=1)
    y = y.reshape(-1, 1)
    coeffecient = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffecient


def plot(eval_result, title: str = None, idx: int = 1):
    ax = fig.add_subplot(int(f"{grid_layout}{idx}"), projection='3d')
    ax.set_title(title)
    ax.scatter(Ks, Ps, eval_result, c='r', marker='o')
    X = np.array(list(zip(Ks, Ps)))
    y = np.array(eval_result)
    fit = best_fit_line(X, y)
    # plot plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                       np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]
    ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


plot(name_vi, "name_vi", 1)
plot(tv_vi, "tv_vi", 2)
plot(name_purity, "name_purity",3)
plot(tv_purity, "tv_purity", 4)
plt.show()
