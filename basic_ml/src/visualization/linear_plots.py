import matplotlib.pyplot as plt


def make_linear_plot(accuracies, sizes, title="", line_color="#2A6EA6", dot_color="#FFA933", xscale="linear"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sizes, accuracies, color=line_color)
    ax.plot(sizes, accuracies, "o", color=dot_color)
    ax.set_xlim(sizes[0], sizes[-1])
    ax.set_ylim(50, 100)
    ax.set_xscale(xscale)
    ax.grid(True)
    ax.set_xlabel("Training set size")
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    acc = [70, 78, 83, 89, 92.3, 93.1, 93.5, 94.1, 94.7]
    make_linear_plot(acc, SIZES, title="Accuracy (%) on the validation data")
    make_linear_plot(acc, SIZES, title="Accuracy (%) on the validation data", xscale="log")
