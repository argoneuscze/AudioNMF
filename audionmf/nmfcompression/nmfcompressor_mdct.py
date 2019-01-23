import struct

import nimfa
import numpy

from audionmf.audio.channel import Channel
from audionmf.transforms.mdct import mdct, imdct
from audionmf.util.matrix_util import serialize_matrix, deserialize_matrix, increment_by_min, matrix_split
from audionmf.util.plot_util import plot_signal


class NMFCompressorMDCT:
    # amount of samples per frame, must be even
    # as a result, the amount of MDCT ranges will be equal to FRAME_SIZE // 2
    FRAME_SIZE = 1152

    # how many frames to put together in a matrix
    # e.g. 1152 // 2 = 576 subbands (bins), NMF_CHUNK_SIZE = 200 => 200x576 matrix as input to NMF
    NMF_CHUNK_SIZE = 500

    # how many iterations and target rank of NMF
    NMF_MAX_ITER = 500
    NMF_RANK = 40

    def compress(self, audio_data, f):
        print('Compressing...')

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for i, channel in enumerate(audio_data.channels):
            # find the resulting MDCT for the entire signal
            mdct_matrix, padding = mdct(channel.samples, self.FRAME_SIZE // 2)

            print(mdct_matrix)

            # write padding
            f.write(struct.pack('<I', padding))

            # increment the MDCT matrix to make sure it's positive
            mdct_matrix_inc, min_val = increment_by_min(mdct_matrix)

            # write minimum value to be subtracted later
            f.write(struct.pack('<d', min_val))

            # split the matrix into chunks
            submatrices = matrix_split(mdct_matrix_inc, self.NMF_CHUNK_SIZE)

            # write the chunk count into the file
            f.write(struct.pack('<I', len(submatrices)))

            # run NMF on the MDCT matrices, getting their weights and coefficients
            for submatrix in submatrices:
                nmf = nimfa.Nmf(submatrix, max_iter=self.NMF_MAX_ITER, rank=self.NMF_RANK,
                                objective='div', update='divergence')()

                W = nmf.basis()
                H = nmf.coef()

                # write both matrices into the file
                serialize_matrix(f, W)
                serialize_matrix(f, H)

            # debug stuff
            plot_signal(channel.samples[:self.FRAME_SIZE // 2], 'dbg_c{}_1_signal.png'.format(i))
            plot_signal(mdct_matrix[2], 'dbg_c{}_2_mdct.png'.format(i))

    def decompress(self, f, audio_data):
        print('Decompressing...')

        data = f.read(4)
        if data != b'ANMF':
            raise Exception('Invalid file format. Expected .anmf.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for i in range(channel_count):
            channel = Channel()

            # read padding
            padding = struct.unpack('<I', f.read(4))[0]

            # read minimum value
            min_val = struct.unpack('<d', f.read(8))[0]

            # read chunk count
            chunk_count = struct.unpack('<I', f.read(4))[0]

            # read and multiply NMF chunks
            chunks = list()

            for _ in range(chunk_count):
                # read both NMF matrices
                W = deserialize_matrix(f)
                H = deserialize_matrix(f)

                # get the original matrix
                mdct_matrix = numpy.matmul(W, H) - min_val

                # add it to the chunk list to be re-joined
                chunks.append(mdct_matrix)

            # re-join the submatrices
            mdct_matrix = numpy.concatenate(chunks)

            # invert MDCT
            signal = imdct(mdct_matrix, padding)

            # convert back to 16-bit signed
            signal = signal.astype(numpy.int16)

            # add samples to channel and finalize
            channel.add_sample_array(signal)
            audio_data.add_channel(channel)

            # debug stuff
            plot_signal(signal[:self.FRAME_SIZE // 2], 'dbg_c{}_2_signal.png'.format(i))
