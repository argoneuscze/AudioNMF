import struct

import nimfa
import numpy

from audionmf.audio.channel import Channel
from audionmf.nmfcompression.matrix_util import array_pad_split, serialize_matrix, deserialize_matrix, array_to_fft, \
    fft_to_array


class NMFCompressor:
    """
    .anmf data format spec

    all multi-byte values are in little endian

    offset | size | description
    8 byte header
    0        4      'ANMF' string
    4        2      # of channels, 16-bit unsigned integer
    6        4      sample rate, 32-bit unsigned integer

    the rest is per-channel data in the following format (relative offsets)
    0        4              padding (extra values to be stripped after multiplication), 32-bit unsigned integer
    4        4              # of rows (W) [r1], 32-bit unsigned integer
    8        4              # of columns (W) [c1], 32-bit unsigned integer
    12       4              # of rows (H) [r2], 32-bit unsigned integer
    16       4              # of columns (H) [c2], 32-bit unsigned integer
    20       r1*c1 + r2*c2  data (W and then H), row by row, 64-bit floats # TODO use 32-bit floats?

    note the data stored is unsigned integers, after multiplication it has to be
    converted back to signed integers

    """

    # bin count = FFT_SIZE / 2
    # resolution = (sampling_rate / 2) / bin_count = Hz per bin up to sampling_rate / 2
    # this number must be even
    FFT_SIZE = 1152

    # how many chunks of FFT to group up together
    # e.g. 576 bins, ARRAY_SIZE = 200 => 576x200 before NMF
    # if set to None it will process all the chunks at once
    ARRAY_SIZE = None

    # how many iterations and target rank of NMF
    NMF_MAX_ITER = 10
    NMF_RANK = 30

    def compress(self, audio_data, output_fd):
        # TODO rewrite
        f = output_fd

        # debug
        # self.FFT_SIZE = 10
        # temp_c = Channel()
        # temp_c.add_sample_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        # audio_data.channels = [temp_c]

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for channel in audio_data.channels:
            # split samples into equal parts
            samples, padding = array_pad_split(channel.samples, self.FFT_SIZE)

            # build two sets of matrices - one with real coefficients and one with complex
            matrices_real = list()
            matrices_imag = list()  # todo later

            # create initial arrays
            matrix_r = numpy.zeros(shape=(len(samples), self.FFT_SIZE // 2))
            matrix_c = numpy.zeros(shape=(len(samples), self.FFT_SIZE // 2))

            # run FFT on each part
            for i, sample_part in enumerate(samples):
                # assign the values to the matrices
                matrix_r[i], matrix_c[i] = array_to_fft(sample_part)

            # increment both matrices by their minimum value so that there aren't any negative values
            r_min = abs(numpy.amin(matrix_r))
            matrix_r += r_min

            c_min = abs(numpy.amin(matrix_c))
            matrix_c += c_min

            # write padding and minimum values to be subtracted later
            f.write(struct.pack('<Idd', padding, r_min, c_min))

            # run NMF on both matrices
            nmf_r = nimfa.Nmf(matrix_r, max_iter=self.NMF_MAX_ITER, rank=self.NMF_RANK)()
            Wr = nmf_r.basis()
            Hr = nmf_r.coef()

            nmf_c = nimfa.Nmf(matrix_c, max_iter=self.NMF_MAX_ITER, rank=self.NMF_RANK)()
            Wc = nmf_c.basis()
            Hc = nmf_c.coef()

            # write both (all 4) matrices into the file
            serialize_matrix(f, Wr)
            serialize_matrix(f, Hr)

            serialize_matrix(f, Wc)
            serialize_matrix(f, Hc)

    def decompress(self, input_fd, audio_data):
        # TODO rewrite
        f = input_fd
        print('====')
        data = f.read(4)
        if data != b'ANMF':
            raise Exception('Invalid file format. Expected .anmf.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for _ in range(channel_count):
            channel = Channel()

            # read information about channel
            padding, r_min, c_min = struct.unpack('<Idd', f.read(20))

            # read matrices
            Wr = deserialize_matrix(f)
            Hr = deserialize_matrix(f)

            Wc = deserialize_matrix(f)
            Hc = deserialize_matrix(f)

            # multiply matrices and subtract old min values
            matrix_r = numpy.matmul(Wr, Hr) - r_min
            matrix_c = numpy.matmul(Wc, Hc) - c_min

            # join the matrices back together
            fft_matrix = matrix_r + matrix_c * 1j

            # iterate over each row and run inverse FFT
            samples = numpy.zeros(shape=(self.FFT_SIZE * fft_matrix.shape[0]))
            for i in range(fft_matrix.shape[0]):
                # run inverse FFT
                ifft = fft_to_array(fft_matrix[i].real, fft_matrix[i].imag)
                samples[i * self.FFT_SIZE:(i + 1) * self.FFT_SIZE] = ifft

            # remove padding and convert back to 16-bit signed integers
            samples = samples[:-padding].astype(numpy.int16)

            print(samples)

            # add samples to channel
            channel.add_sample_array(samples)
            audio_data.add_channel(channel)
