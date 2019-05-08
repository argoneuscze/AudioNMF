# Exploring usage of non-negative matrix factorization in audio compression

This repository stores the implementation part of my Master's thesis.

The text part is visible [**here**](https://github.com/argoneuscze/thesis).

## About

AudioNMF is a tool used to compress audio files in WAV format using non-negative matrix factorization.
It's written entirely in Python 3 and is published under the [MIT](LICENSE.txt) license.

There are three modes of operation:

* *ANMF-RAW* - direct compression of samples
* *ANMF-MDCT* - compression of the MDCT spectrogram
* *ANMF-STFT* - compression of the STFT spectrogram

In practice, however, only **ANMF-STFT** produces practical results.

## Installation

AudioNMF requires **Python 3** (3.5+ preferred) and can be installed by simply using:

`python setup.py install`

And to run the tests:

`python setup.py test`

**Note 1:** To ensure smooth installation, please make sure you have the latest versions
of Python build tools, e.g.:

`python -m pip install --upgrade pip setuptools wheel`

**Note 2:** In case of issues when using Windows, it's recommended to install a pre-made
scientific Python distribution, for example [Anaconda](https://www.anaconda.com/download/).

## Usage

After installation, simply run `audionmf`.
The only currently accepted filetype is a 16-bit signed integer WAV, ideally with a sampling rate of 44.1 KHz.

To convert a WAV file to the accepted format, you can use e.g. `sox`:

`sox input.wav -b 16 -r 44.1k input_16b.wav`

To compress a file:

`audionmf compress -c anmfs input.wav output.anmfs`

To decompress:

`audionmf decompress output.anmfs original.wav`

The application can give you the possible options and arguments using `--help`.
