import struct

import nimfa
import numpy

from audionmf.audio.channel import Channel


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

    def compress(self, audio_data, output_fd):
        # TODO rewrite
        f = output_fd

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for matrix, padding in [ch.to_positive_matrix() for ch in audio_data.channels]:
            # TODO separate into another file?

            # compute NMF
            nmf = nimfa.Nmf(matrix, max_iter=500, rank=50)
            nmf_res = nmf()
            W = nmf_res.basis()
            H = nmf_res.coef()

            # write both matrices to file
            f.write(struct.pack('<IIIII', padding, W.shape[0], W.shape[1], H.shape[0], H.shape[1]))

            f.write(struct.pack('<' + 'd' * W.size, *W.flat))
            f.write(struct.pack('<' + 'd' * H.size, *H.flat))

    def decompress(self, input_fd, audio_data):
        # TODO rewrite
        f = input_fd

        data = f.read(4)
        if data != b'ANMF':
            raise Exception('Invalid file format. Expected .anmf.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for _ in range(channel_count):
            channel = Channel()

            # read information about channel
            padding, wX, wY, hX, hY = struct.unpack('<IIIII', f.read(20))
            wSize = wX * wY
            hSize = hX * hY

            # read matrix
            # todo Channel.from_positive_matrix()
            dt = numpy.dtype(numpy.float64)
            dt = dt.newbyteorder('<')
            wRaw = numpy.frombuffer(f.read(wSize * 8), dtype=dt)
            hRaw = numpy.frombuffer(f.read(hSize * 8), dtype=dt)

            W = numpy.reshape(wRaw, (wX, wY))
            H = numpy.reshape(hRaw, (hX, hY))

            # remove padding, decrement and convert back to 16-bit signed integers
            channel_data = numpy.matmul(W, H).reshape(-1)[:-padding] - 2 ** 15
            channel_data = channel_data.astype(numpy.int16)

            channel.add_sample_array(channel_data)
            audio_data.add_channel(channel)
