import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def plot_line(x_values, y_values, m, c):
    _ = plt.plot(x_values, y_values, "o", label="Original data", markersize=10)
    _ = plt.plot(x_values, m * x_values + c, "r", label="Fitted line")
    _ = plt.legend()
    _ = plt.title(f"intercept: {c:.2f}; slope: {m:.2f}")
    plt.show()


def make_linear_plot(
    accuracies,
    sizes,
    title="",
    line_color="#2A6EA6",
    dot_color="#FFA933",
    xscale="linear",
):
    fig = plt.figure()
    sub_plot = fig.add_subplot(111)
    sub_plot.plot(sizes, accuracies, color=line_color)
    sub_plot.plot(sizes, accuracies, "o", color=dot_color)
    sub_plot.set_xlim(sizes[0], sizes[-1])
    sub_plot.set_ylim(50, 100)
    sub_plot.set_xscale(xscale)
    sub_plot.grid(True)
    sub_plot.set_xlabel("Training set size")
    sub_plot.set_title(title)
    plt.show()


def voronoi_tessellation(data_points: np.ndarray):
    vor = Voronoi(data_points)
    _ = voronoi_plot_2d(vor,
                          show_vertices=False,
                          line_colors='orange',
                          line_width=2,
                          line_alpha=0.6,
                          point_size=2)
    plt.show()


if __name__ == "__main__":
    acc = [70, 78, 83, 89, 92.3, 93.1, 93.5, 94.1, 94.7]
    make_linear_plot(acc, SIZES, title="Accuracy (%) on the validation data")
    make_linear_plot(acc, SIZES, title="Accuracy (%) on the validation data", xscale="log")

    rng = np.random.default_rng()
    points = rng.random((10, 2))
    voronoi_tessellation(points)