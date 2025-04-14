import matplotlib.pyplot as plt
import numpy as np

def plot_paths(planets, paths, best_path=None, title='Bee paths'):
    fig, ax = plt.subplots()
    # planets
    colors = dict(Sun='yellow', Start='green', Obstacle='red', Target='blue')
    for p in planets:
        ax.scatter(*p.pos, s=100, color=colors.get(p.name, 'black'), label=p.name, zorder=3)
    # all bee paths
    for path in paths:
        pts = np.vstack(path)
        ax.plot(pts[:,0], pts[:,1], lw=0.5, color='grey', alpha=0.6)
    # best
    if best_path is not None:
        pts = np.vstack(best_path)
        ax.plot(pts[:,0], pts[:,1], lw=2.5, color='purple', label='best')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    plt.show()
