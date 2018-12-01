import math
import struct

import nimfa
import numpy

from audionmf.audio.channel import Channel
from audionmf.nmfcompression.matrix_util import array_pad_split, serialize_matrix, deserialize_matrix, increment_by_min


class NMFCompressorRaw:
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

    # tuple that says how large chunks to group up together into matrices
    # (rows, cols), will be padded with zeros if too small
    # if set to None, the whole signal will be one chunk with a square size
    CHUNK_SHAPE = (1000, 150)

    # how many iterations and target rank of NMF
    NMF_MAX_ITER = 500
    NMF_RANK = 30

    def compress(self, audio_data, output_fd):
        f = output_fd

        # debug
        # self.FFT_SIZE = 10
        # temp_c = Channel()
        # temp_c.add_sample_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        # audio_data.channels = [temp_c]

        print('Compressing...')

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for channel in audio_data.channels:
            # determine chunk size
            if self.CHUNK_SHAPE is not None:
                chunk_size = self.CHUNK_SHAPE[0] * self.CHUNK_SHAPE[1]
            else:
                sample_cnt = len(channel.samples)
                square_dim = math.ceil(math.sqrt(sample_cnt))
                chunk_size = square_dim ** 2
                self.CHUNK_SHAPE = (square_dim, square_dim)

            # split samples into equal parts
            samples, padding = array_pad_split(channel.samples, chunk_size)

            # create initial matrices
            matrix_list = list()

            # re-shape each part into a matrix of the given shape and convert to a larger datatype
            for sample_part in samples:
                sample_part_matrix = numpy.reshape(sample_part, self.CHUNK_SHAPE).astype(numpy.int32)
                matrix_list.append(sample_part_matrix)

            # write padding of samples after decompression and the amount of matrices / 2
            # (there's two matrices per matrix due to NMF)
            f.write(struct.pack('<II', padding, len(matrix_list)))

            # process all the matrices, real ones first
            for matrix in matrix_list:
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
        f = input_fd

        print('Decompressing...')

        data = f.read(4)
        if data != b'ANMF':
            raise Exception('Invalid file format. Expected .anmf.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for _ in range(channel_count):
            channel = Channel()

            # read padding and amount of matrices / 2
            padding, matrix_count = struct.unpack('<II', f.read(8))

            array_list = list()

            # there's 2 times more matrices, see above
            for i in range(matrix_count):
                # read minimum value
                min_val = struct.unpack('<d', f.read(8))

                # read both NMF matrices
                W = deserialize_matrix(f)
                H = deserialize_matrix(f)

                # multiply matrices and subtract old min values
                matrix = numpy.matmul(W, H) - min_val

                # turn the matrix back into an array
                ary = matrix.ravel()

                # add to list
                array_list.append(ary)

            # turn the array list into one large array
            samples = numpy.concatenate(array_list, 0)

            # remove padding and convert back to 16-bit signed integers
            samples = samples[:-padding].astype(numpy.int16)

            # add samples to channel
            channel.add_sample_array(samples)
            audio_data.add_channel(channel)
