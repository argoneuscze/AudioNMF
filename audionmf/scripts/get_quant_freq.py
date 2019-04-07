from pprint import pprint

import matplotlib.pyplot as plt
import numpy

freq_list = []
frequencies = {}


def increment_frequency(value):
    global frequencies
    if value not in frequencies:
        frequencies[value] = 0
    frequencies[value] += 1


def freq_done():
    global frequencies
    global freq_list
    freq_list.append(frequencies)
    frequencies = {}


def plot_frequencies():
    # find the average of the frequencies
    avg_freq = {}
    levels = len(freq_list[0])

    for i in range(levels):
        try:
            avg_freq[i] = int(numpy.round(numpy.mean([freq_dict[i] for freq_dict in freq_list])))
        except KeyError:
            avg_freq[i] = 0

    # print values
    pprint(avg_freq)

    # plot a histogram
    data = sorted(avg_freq.items())
    x, y = zip(*data)
    fig, ax = plt.subplots()
    ax.set_title('Frequency of quantized values')
    ax.set_xlabel('Quantized value')
    ax.set_ylabel('Frequency')
    ax.bar(x, y)
    fig.tight_layout()
    fig.savefig('frequency_data.png')
    plt.close(fig)
