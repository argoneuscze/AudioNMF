import matplotlib.pyplot as plt


def plot_function(func, min_val, max_val, step, filename, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    ax.set_xticks(range(1, 25))
    ax.set_xlim(1, 24)
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    val_range = range(min_val, max_val, step)
    data = [func(x) for x in val_range]
    ax.plot(data, val_range, marker='o')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


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


def plot_spectrogram(signal, filename, title='Spectrogram', x_label='Time (s)', y_label='Frequency (Hz)', rate=44100,
                     size=1152, overlap=576):
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.specgram(signal, Fs=rate, NFFT=size, noverlap=overlap, mode='psd', scale='dB')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
