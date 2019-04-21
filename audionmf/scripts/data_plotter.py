import matplotlib.pyplot as plt

data = []


def add_data_point(value):
    global data
    data.append(value)


def plot_data(x_label, y_label):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(data)
    fig.tight_layout()
    fig.savefig('data_plot.png')
    plt.close(fig)
