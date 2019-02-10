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
        print('Compressing...')

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for i, channel in enumerate(audio_data.channels):
            stft = scipy.signal.stft(channel.samples, nperseg=self.FRAME_SIZE, padded=True)[2]

            # debug
            freq = numpy.absolute(stft[:, 200])[:100]
            plot_signal(freq, "dbg_signal_1.png")

            # transpose for consistency with other methods
            stft = numpy.transpose(stft)

            # split the matrix into chunks
            submatrices = matrix_split(stft, self.NMF_CHUNK_SIZE)

            # write the chunk count into the file
            f.write(struct.pack('<I', len(submatrices)))

            # run NMF on the FFT matrices, getting their weights and coefficients
            for submatrix in submatrices:
                # split real and imaginary part
                sub_r = submatrix.real
                sub_i = submatrix.imag

                # run NMF
                W_r, H_r, min_r = nmf_matrix(sub_r, self.NMF_MAX_ITER, self.NMF_RANK)
                W_i, H_i, min_i = nmf_matrix(sub_i, self.NMF_MAX_ITER, self.NMF_RANK)

                # write minimum value to be subtracted later
                f.write(struct.pack('<dd', min_r, min_i))

                # write matrices to file
                serialize_matrix(f, W_r)
                serialize_matrix(f, H_r)

                serialize_matrix(f, W_i)
                serialize_matrix(f, H_i)

    def decompress(self, f, audio_data):
        print('Decompressing...')

        data = f.read(4)
        if data != b'ANMF':
            raise Exception('Invalid file format. Expected .anmf.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for i in range(channel_count):
            channel = Channel()

            # read chunk count
            chunk_count = struct.unpack('<I', f.read(4))[0]

            # read and multiply NMF chunks
            chunks = list()

            for _ in range(chunk_count):
                # read minimum values
                min_r, min_i = struct.unpack('<dd', f.read(16))

                # read NMF matrices
                W_r = deserialize_matrix(f)
                H_r = deserialize_matrix(f)

                W_i = deserialize_matrix(f)
                H_i = deserialize_matrix(f)

                # get original chunk back
                stft_chunk_r = nmf_matrix_original(W_r, H_r, min_r)
                stft_chunk_i = nmf_matrix_original(W_i, H_i, min_i)
                stft_chunk = stft_chunk_r + 1j * stft_chunk_i

                # append it to the list
                chunks.append(stft_chunk)

            # concatenate STFT chunks
            stft = numpy.concatenate(chunks)

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
            plot_signal(freq, "dbg_signal_2.png")
