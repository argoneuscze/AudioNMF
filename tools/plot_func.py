import matplotlib.pyplot as plt
import numpy


def bitrate_rank(r):
    return (r * (32 * 500 + 0.5 * 1152 * 2.952) * (44100 / (0.5 * 1152 * 500))) / 1000


def bitrate_chunk(c):
    return (40 * (32 * c + 0.5 * 1152 * 2.952) * (44100 / (0.5 * 1152 * c))) / 1000


def plot_function(func, min_val, max_val, step, filename, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    val_range = numpy.arange(min_val, max_val, step)
    data = [func(x) for x in val_range]
    ax.plot(val_range, data)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


plot_function(bitrate_rank, 20, 100, 0.1, "bitrate_rank.png", x_label='NMF Rank', y_label='Bitrate (kbps)')
plot_function(bitrate_chunk, 50, 800, 0.1, "bitrate_chunk.png", x_label='Chunk size', y_label='Bitrate (kbps)')
