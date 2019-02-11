import struct

import numpy
import scipy.signal

from audionmf.audio.channel import Channel
from audionmf.util.matrix_util import serialize_matrix, deserialize_matrix, matrix_split
from audionmf.util.nmf_util import nmf_matrix, nmf_matrix_original
from audionmf.util.plot_util import plot_signal


class NMFCompressorSTFT:
    # amount of samples per frame, must be even
    FRAME_SIZE = 1152

    # how many frames to put together in a matrix
    # e.g. (1152 // 2) + 1 = 577 subbands (bins), NMF_CHUNK_SIZE = 200 => 577x200 matrix as input to NMF
    NMF_CHUNK_SIZE = 500

    # how many iterations and target rank of NMF
    NMF_MAX_ITER = 200
    NMF_RANK = 30

    def compress(self, audio_data, f):
        print('Compressing (STFT)...')

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for i, channel in enumerate(audio_data.channels):
            stft = scipy.signal.stft(channel.samples, nperseg=self.FRAME_SIZE, padded=True)[2]

            # debug
            freq = numpy.absolute(stft[:, 200])[:100]
            plot_signal(freq, "dbg_signal_1a.png")

            # transpose for consistency with other methods
            stft = numpy.transpose(stft)

            # find phase and magnitude matrices
            phases = numpy.angle(stft)
            magnitudes = numpy.absolute(stft)

            # serialize the phases matrix
            serialize_matrix(f, phases)

            # split the magnitude matrix into chunks
            submatrices = matrix_split(magnitudes, self.NMF_CHUNK_SIZE)

            # write the chunk count into the file
            f.write(struct.pack('<I', len(submatrices)))

            # run NMF on the magnitude submatrices, getting their weights and coefficients
            for submatrix in submatrices:
                W, H, min_val = nmf_matrix(submatrix, self.NMF_MAX_ITER, self.NMF_RANK)

                # write minimum value to be subtracted later
                f.write(struct.pack('<d', min_val))

                # write matrices to file
                serialize_matrix(f, W)
                serialize_matrix(f, H)

    def decompress(self, f, audio_data):
        print('Decompressing (STFT)...')

        data = f.read(4)
        if data != b'ANMF':
            raise Exception('Invalid file format. Expected .anmf.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for i in range(channel_count):
            channel = Channel()

            # read phases matrix
            phases = deserialize_matrix(f)

            # read chunk count
            chunk_count = struct.unpack('<I', f.read(4))[0]

            # read and multiply NMF chunks
            chunks = list()

            for _ in range(chunk_count):
                # read minimum value
                min_val = struct.unpack('<d', f.read(8))

                # read NMF magnitude matrices
                W = deserialize_matrix(f)
                H = deserialize_matrix(f)

                # get original chunk back
                mag_chunk = nmf_matrix_original(W, H, min_val)

                # append it to the list
                chunks.append(mag_chunk)

            # concatenate magnitude matrix chunks
            magnitudes = numpy.concatenate(chunks)

            # join matrices back into the original STFT matrix
            stft = magnitudes * numpy.cos(phases) + 1j * magnitudes * numpy.sin(phases)

            # transpose back into original form
            stft = numpy.transpose(stft)

            # run inverse STFT
            signal = scipy.signal.istft(stft, nperseg=self.FRAME_SIZE)[1]

            # convert back to 16-bit signed
            signal = signal.astype(numpy.int16)

            # add samples to channel and finalize
            channel.add_sample_array(signal)
            audio_data.add_channel(channel)

            # debug
            freq = numpy.absolute(stft[:, 200])[:100]
            plot_signal(freq, "dbg_signal_2a.png")
