import struct

import numpy
from scipy import fftpack

from audionmf.audio.channel import Channel
from audionmf.nmfcompression.matrix_util import array_pad, serialize_matrix, deserialize_matrix


class NMFCompressorMDCT:
    # amount of samples per frame, must be even
    # as a result, the amount of MDCT ranges will be equal to FRAME_SIZE // 2
    FRAME_SIZE = 1152

    # how many frames to put together in a matrix
    # e.g. 1152 // 2 = 576 subbands (bins), NMF_CHUNK_SIZE = 200 => 200x576 matrix as input to NMF
    NMF_CHUNK_SIZE = 100

    # how many iterations and target rank of NMF
    NMF_MAX_ITER = 100
    NMF_RANK = 30

    def compress(self, audio_data, f):
        print('Compressing...')

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for channel in audio_data.channels:
            # pad samples properly to 2x frame size
            samples, padding = array_pad(channel.samples, self.FRAME_SIZE * 2)

            # MDCT
            # allocate MDCT output matrix
            mdct = numpy.ndarray((len(samples) // self.FRAME_SIZE, self.FRAME_SIZE))

            # size of one input block, equal to N // 2
            block_size = self.FRAME_SIZE // 2
            for i in range(len(samples) // self.FRAME_SIZE - 1):
                # where the current frame starts
                frame_start = i * self.FRAME_SIZE

                # create blocks for MDCT
                # the MDCT of 2N inputs (a, b, c, d) is exactly equivalent to a DCT-IV of the N inputs: (−cR−d, a−bR)
                a = samples[frame_start:frame_start + block_size]
                bR = samples[frame_start + block_size:frame_start + 2 * block_size][::-1]
                cR = samples[frame_start + 2 * block_size:frame_start + 3 * block_size][::-1]
                d = samples[frame_start + 3 * block_size:frame_start + 4 * block_size]

                # fill input array
                dct_input = numpy.concatenate((-cR - d, a - bR))

                # run DCT-IV on this array of size N, producing effectively MDCT of size 2N
                dct4 = fftpack.dct(dct_input, type=4)

                # emplace the resulting MDCT in the output matrix
                mdct[i] = dct4

            # write padding
            f.write(struct.pack('<I', padding))

            # write the MDCT matrix to file
            serialize_matrix(f, mdct)

            # TODO NMF
            # TODO filter bank?
            # TODO windowing?

    def decompress(self, f, audio_data):
        print('Decompressing...')

        data = f.read(4)
        if data != b'ANMF':
            raise Exception('Invalid file format. Expected .anmf.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for _ in range(channel_count):
            channel = Channel()

            # read padding
            padding = struct.unpack('<I', f.read(4))[0]

            # read MDCT matrix
            mdct = deserialize_matrix(f)

            # invert MDCT
            # TODO
