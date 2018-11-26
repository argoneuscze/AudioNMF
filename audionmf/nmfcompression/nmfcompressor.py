import struct

import nimfa
import numpy

from audionmf.audio.channel import Channel
from audionmf.nmfcompression.matrix_util import array_pad_split, serialize_matrix, deserialize_matrix, array_to_fft, \
    fft_to_array, increment_by_min


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
            matrices_imag = list()

            # find actual chunk size
            chunk_size = len(samples)
            if self.ARRAY_SIZE is not None:
                if self.ARRAY_SIZE < chunk_size:
                    chunk_size = self.ARRAY_SIZE

            # create initial matrices
            matrix_r = matrix_c = None

            # run FFT on each part and save each chunk
            i = 0
            for sample_part in samples:
                # check if we should make new matrices
                if i % chunk_size == 0:
                    # add existing matrices to the lists
                    if matrix_r is not None:
                        matrices_real.append(matrix_r)
                        matrices_imag.append(matrix_c)

                    # update to lower chunk size if close to the end to save a bit of space
                    remaining = len(samples) - len(matrices_real) * chunk_size
                    if remaining < chunk_size:
                        chunk_size = remaining

                    # create new matrices
                    matrix_r = numpy.zeros(shape=(chunk_size, self.FFT_SIZE // 2))
                    matrix_c = numpy.zeros(shape=(chunk_size, self.FFT_SIZE // 2))

                    # reset index
                    i = 0

                # assign the values to the matrices
                matrix_r[i], matrix_c[i] = array_to_fft(sample_part)

                # increment index
                i += 1

            # add the left-over matrices
            matrices_real.append(matrix_r)
            matrices_imag.append(matrix_c)
            # TODO split matrices out of a large one

            # write padding of samples after decompression and the amount of matrices / 4
            # (there's imaginary and real matrices, we only write down the count of the real ones,
            # which are decomposed via NMF)
            f.write(struct.pack('<II', padding, len(matrices_real)))

            # process all the matrices, real ones first
            for matrix in matrices_real + matrices_imag:
                # increment all matrices by their minimum value so that there aren't any negative values
                matrix, min_val = increment_by_min(matrix)

                # write minimum value to be subtracted later
                f.write(struct.pack('<d', min_val))

                # run NMF on the matrix, getting its weights and coefficients
                nmf = nimfa.Nmf(matrix, max_iter=self.NMF_MAX_ITER, rank=self.NMF_RANK)()
                W = nmf.basis()
                H = nmf.coef()

                # write both matrices into the file
                serialize_matrix(f, W)
                serialize_matrix(f, H)

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

            # read padding and amount of matrices / 4
            padding, real_matrix_count = struct.unpack('<II', f.read(8))

            all_matrices = list()

            # there's 4 times more matrices, see above
            for i in range(real_matrix_count * 2):
                # read minimum value
                min_val = struct.unpack('<d', f.read(8))

                # read both NMF matrices
                W = deserialize_matrix(f)
                H = deserialize_matrix(f)

                # multiply matrices and subtract old min values
                matrix = numpy.matmul(W, H) - min_val

                # add to list
                all_matrices.append(matrix)

            matrix_real = numpy.concatenate(all_matrices[:len(all_matrices) // 2], 0)
            matrix_imag = numpy.concatenate(all_matrices[len(all_matrices) // 2:], 0)

            # join the matrices back together
            fft_matrix = matrix_real + matrix_imag * 1j

            # iterate over each row and run inverse FFT
            samples = numpy.zeros(shape=(self.FFT_SIZE * fft_matrix.shape[0]))
            for i in range(fft_matrix.shape[0]):
                # run inverse FFT
                ifft = fft_to_array(fft_matrix[i].real, fft_matrix[i].imag)
                samples[i * self.FFT_SIZE:(i + 1) * self.FFT_SIZE] = ifft

            # remove padding and convert back to 16-bit signed integers
            samples = samples[:-padding].astype(numpy.int16)

            # add samples to channel
            channel.add_sample_array(samples)
            audio_data.add_channel(channel)
