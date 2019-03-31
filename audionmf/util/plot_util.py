import matplotlib.pyplot as plt


def plot_signal(signal, filename, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.plot(signal)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
