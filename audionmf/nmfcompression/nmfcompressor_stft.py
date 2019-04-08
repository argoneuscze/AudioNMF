import struct

import numpy
import scipy.signal

from audionmf.audio.channel import Channel
from audionmf.transforms.huffman import HuffmanCoder
from audionmf.transforms.quantization import scale_val, mu_law_compand, mu_law_expand, UniformQuantizer
from audionmf.util.matrix_util import serialize_matrix, deserialize_matrix, matrix_split
from audionmf.util.nmf_util import nmf_matrix, nmf_matrix_original


class NMFCompressorSTFT:
    # amount of samples per frame, must be even
    # 1152 frame size at 44100 sample rate corresponds to ~26 ms windows
    FRAME_SIZE = 1152

    # how many frames to put together in a matrix
    # e.g. (1152 // 2) + 1 = 577 subbands (bins), NMF_CHUNK_SIZE = 200 => 577x200 matrix as input to NMF
    NMF_CHUNK_SIZE = 500

    # how many iterations and target rank of NMF
    NMF_MAX_ITER = 200
    NMF_RANK = 30

    def __init__(self):
        # initialize Huffman encoder/decoder
        self.huffman = HuffmanCoder('stft32')
        self.quantizer = UniformQuantizer(0, 1, 32)
        self.quantize_vec = numpy.vectorize(self.quantizer.quantize_value)
        self.dequantize_vec = numpy.vectorize(self.quantizer.dequantize_index)
        self.scale_matrix = numpy.vectorize(scale_val)
        self.compand = numpy.vectorize(mu_law_compand)
        self.expand = numpy.vectorize(mu_law_expand)

    def compress(self, audio_data, f):
        print('Compressing (STFT)...')

        f.write(b'ANMF')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for i, channel in enumerate(audio_data.channels):
            stft = scipy.signal.stft(channel.samples, fs=audio_data.sample_rate, window='hann',
                                     noverlap=self.FRAME_SIZE // 2, nperseg=self.FRAME_SIZE, padded=True)[2]

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
                # run NMF on the matrix
                W, H, min_val = nmf_matrix(submatrix, self.NMF_MAX_ITER, self.NMF_RANK)

                # scale values to [0,1] using the maximum range of both matrices
                matrix_min = min(numpy.amin(W), numpy.amin(H))
                matrix_max = max(numpy.amax(W), numpy.amax(H))

                Ws = self.scale_matrix(W, matrix_min, matrix_max, 0, 1)
                Hs = self.scale_matrix(H, matrix_min, matrix_max, 0, 1)

                # compand the scaled matrices using mu-law
                Wsc = self.compand(Ws, 10 ** 4)
                Hsc = self.compand(Hs, 10 ** 5)

                # uniformly quantize the mu-law scaled matrix B (basis)
                # 32 levels of quantization between <0,1>
                Wscq = self.quantize_vec(Wsc)

                # TODO remove debug
                # for val in numpy.nditer(Wscq):
                # increment_frequency(int(val))
                # for val in numpy.nditer(Hscq):
                #    increment_frequency(int(val))
                # freq_done()

                # Huffman encode the matrix
                Wout, Wrows = self.huffman.encode_int_matrix(Wscq)

                # now write everything to file

                # write minimum value to be subtracted later
                f.write(struct.pack('<d', min_val))

                # write the min and max to be re-scaled later
                f.write(struct.pack('<dd', matrix_min, matrix_max))

                # write companded W matrix
                serialize_matrix(f, Hsc)

                # write the quantized matrix H and number of rows
                f.write(struct.pack('<II', Wrows, len(Wout)))
                f.write(Wout)

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

            # read magnitude matrix chunk count
            chunk_count = struct.unpack('<I', f.read(4))[0]

            # read and multiply NMF chunks to obtain magnitude matrix
            chunks = list()

            for _ in range(chunk_count):
                # read minimum value
                min_val = struct.unpack('<d', f.read(8))[0]

                # read min and max for re-scaling
                matrix_min, matrix_max = struct.unpack('<dd', f.read(16))

                # read companded matrix H
                Hsc = deserialize_matrix(f)

                # read Huffman encoded matrix H
                Wrows, Wlen = struct.unpack('<II', f.read(8))
                Wbytes = f.read(Wlen)

                # Huffman decode the matrix to gain quantized values
                Wscq = self.huffman.decode_int_matrix(Wbytes, Wrows)

                # multiply each value by step to gain original values
                Wsc = self.dequantize_vec(Wscq)

                # expand the scaled matrices using mu-law
                Ws = self.expand(Wsc, 10 ** 4)
                Hs = self.expand(Hsc, 10 ** 5)

                # scale matrices back to normal
                W = self.scale_matrix(Ws, 0, 1, matrix_min, matrix_max)
                H = self.scale_matrix(Hs, 0, 1, matrix_min, matrix_max)

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
            signal = scipy.signal.istft(stft, fs=audio_data.sample_rate, window='hann',
                                        noverlap=self.FRAME_SIZE // 2, nperseg=self.FRAME_SIZE, )[1]

            # convert back to 16-bit signed
            signal = signal.astype(numpy.int16)

            # add samples to channel and finalize
            channel.add_sample_array(signal)
            audio_data.add_channel(channel)
