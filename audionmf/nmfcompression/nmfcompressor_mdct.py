import struct

import numpy

from audionmf.audio.channel import Channel
from audionmf.transforms.mdct import mdct, imdct
from audionmf.util.matrix_util import serialize_matrix, deserialize_matrix


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

        for i, channel in enumerate(audio_data.channels):
            # find the resulting MDCT for the entire signal
            mdct_matrix, padding = mdct(channel.samples, self.FRAME_SIZE // 2)

            # write padding
            f.write(struct.pack('<I', padding))

            # write the MDCT matrix to file
            serialize_matrix(f, mdct_matrix)

            # debug stuff
            # plot_signal(channel.samples[:self.FRAME_SIZE // 2], 'dbg_c{}_1_signal.png'.format(i))
            # plot_signal(mdct_matrix[2], 'dbg_c{}_2_mdct.png'.format(i))

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

        for i in range(channel_count):
            channel = Channel()

            # read padding
            padding = struct.unpack('<I', f.read(4))[0]

            # read MDCT matrix
            mdct_matrix = deserialize_matrix(f)

            # invert MDCT
            signal = imdct(mdct_matrix, padding)

            # convert back to 16-bit signed
            signal = signal.astype(numpy.int16)

            # add samples to channel and finalize
            channel.add_sample_array(signal)
            audio_data.add_channel(channel)
