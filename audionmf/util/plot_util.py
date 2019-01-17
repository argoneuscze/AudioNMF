import matplotlib.pyplot as plt


def plot_signal(signal, filename):
    fig, ax = plt.subplots()
    ax.plot(signal)
    fig.savefig(filename)
    plt.close(fig)
