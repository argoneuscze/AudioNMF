import struct

import numpy

from audionmf.audio.channel import Channel
from audionmf.util.matrix_util import array_pad_split, serialize_matrix, deserialize_matrix, array_to_fft, \
    fft_to_array, matrix_split
from audionmf.util.nmf_util import nmf_matrix


class NMFCompressorFFT:
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

    # TODO this whole thing is outdated

    """

    # bin count = FFT_SIZE / 2
    # resolution = (sampling_rate / 2) / bin_count = Hz per bin up to sampling_rate / 2
    # this number must be even
    FFT_SIZE = 1152

    # how many chunks of FFT to group up together
    # e.g. 576 bins, ARRAY_SIZE = 200 => 200x576 before NMF
    # if set to None it will process all the chunks at once
    ARRAY_SIZE = 200

    # how many iterations and target rank of NMF
    NMF_MAX_ITER = 400
    NMF_RANK = 50

    def compress(self, audio_data, output_fd):
        f = output_fd

        print('Compressing (FFT)...')

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for channel in audio_data.channels:
            # split samples into equal parts
            samples, padding = array_pad_split(channel.samples, self.FFT_SIZE)

            # create initial matrices
            matrix_r = numpy.zeros(shape=(len(samples), self.FFT_SIZE // 2))
            matrix_c = numpy.zeros(shape=(len(samples), self.FFT_SIZE // 2))

            # run FFT on each part and save each chunk
            for i, sample_part in enumerate(samples):
                matrix_r[i], matrix_c[i] = array_to_fft(sample_part)

            # build two sets of smaller matrices - one with real coefficients and one with complex
            # and split them into chunks
            matrices_real = matrix_split(matrix_r, self.ARRAY_SIZE)
            matrices_imag = matrix_split(matrix_c, self.ARRAY_SIZE)

            # write padding of samples after decompression and the amount of matrices / 4
            # (there's imaginary and real matrices, we only write down the count of the real ones,
            # which are decomposed via NMF)
            f.write(struct.pack('<II', padding, len(matrices_real)))

            # process all the matrices, real ones first
            for matrix in matrices_real + matrices_imag:
                # run NMF on the matrix
                W, H, min_val = nmf_matrix(matrix, self.NMF_MAX_ITER, self.NMF_RANK)

                # write minimum value to be subtracted later
                f.write(struct.pack('<d', min_val))

                # write both matrices into the file
                serialize_matrix(f, W)
                serialize_matrix(f, H)

    def decompress(self, input_fd, audio_data):
        f = input_fd

        print('Decompressing (FFT)...')

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
