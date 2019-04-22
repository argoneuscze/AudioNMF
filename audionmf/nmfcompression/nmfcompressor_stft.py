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
    NMF_MAX_ITER = 1000
    NMF_RANK = 40

    # mu-law companding parameters
    MU_LAW_W = 10 ** 4
    MU_LAW_H = 10 ** 5

    def __init__(self):
        # initialize Huffman encoder/decoder
        self.Phuffman = HuffmanCoder('stftp')
        self.Pquantizer = UniformQuantizer(-numpy.pi, numpy.pi, 2 ** 3)
        self.Pquantize_vec = numpy.vectorize(self.Pquantizer.quantize_value)
        self.Pdequantize_vec = numpy.vectorize(self.Pquantizer.dequantize_index)
        self.Hhuffman = HuffmanCoder('stft32')
        self.Hquantizer = UniformQuantizer(0, 1, 2 ** 5)
        self.Hquantize_vec = numpy.vectorize(self.Hquantizer.quantize_value)
        self.Hdequantize_vec = numpy.vectorize(self.Hquantizer.dequantize_index)
        self.scale_matrix = numpy.vectorize(scale_val)
        self.compand = numpy.vectorize(mu_law_compand)
        self.expand = numpy.vectorize(mu_law_expand)

    def compress(self, audio_data, f):
        print('Compressing (STFT)...')

        f.write(b'ANMFS')
        f.write(struct.pack('<HI', len(audio_data.channels), audio_data.sample_rate))

        for i, channel in enumerate(audio_data.channels):
            stft = scipy.signal.stft(channel.samples, fs=audio_data.sample_rate, window='hann',
                                     noverlap=self.FRAME_SIZE // 2, nperseg=self.FRAME_SIZE, padded=True)[2]

            # transpose for consistency with other methods
            stft = numpy.transpose(stft)

            # find phase and magnitude matrices
            phases = numpy.angle(stft)
            magnitudes = numpy.absolute(stft)

            # split the magnitude matrix into chunks
            submatrices = matrix_split(magnitudes, self.NMF_CHUNK_SIZE)

            # write the magnitude chunk count into the file
            f.write(struct.pack('<I', len(submatrices)))

            # compress phases using Huffman
            Pq = self.Pquantize_vec(phases)
            Pout, Prows = self.Phuffman.encode_int_matrix(Pq)

            # write quantized phase matrix
            f.write(struct.pack('<II', Prows, len(Pout)))
            f.write(Pout)

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
                Wsc = self.compand(Ws, self.MU_LAW_W)
                Hsc = self.compand(Hs, self.MU_LAW_H)

                # uniformly quantize the mu-law scaled matrix H (coefficients)
                # 32 levels of quantization between <0,1>
                Hscq = self.Hquantize_vec(Hsc)

                # debug
                # for val in numpy.nditer(Wscq):
                # increment_frequency(int(val))
                # for val in numpy.nditer(Hscq):
                #    increment_frequency(int(val))
                # freq_done()

                # Huffman encode the matrix
                Hout, Hrows = self.Hhuffman.encode_int_matrix(Hscq)

                # for W, we scale it to 32-bit unsigned int
                Wscs = self.scale_matrix(Wsc, 0, 1, 0, 2 ** 32).astype(numpy.uint32)

                # now write everything to file

                # write minimum value to be subtracted later
                f.write(struct.pack('<d', min_val))

                # write the min and max to be re-scaled later
                f.write(struct.pack('<dd', matrix_min, matrix_max))

                # write companded scaled W matrix
                serialize_matrix(f, Wscs, 'I')

                # write the quantized matrix H and number of rows
                f.write(struct.pack('<II', Hrows, len(Hout)))
                f.write(Hout)

    def decompress(self, f, audio_data):
        print('Decompressing (STFT)...')

        data = f.read(5)
        if data != b'ANMFS':
            raise Exception('Invalid file format. Expected .anmfs.')
        channel_count, sample_rate = struct.unpack('<HI', f.read(6))
        audio_data.sample_rate = sample_rate

        for i in range(channel_count):
            channel = Channel()

            # read magnitude matrix chunk count
            chunk_count = struct.unpack('<I', f.read(4))[0]

            # read phases matrix
            Prows, Plen = struct.unpack('<II', f.read(8))
            Pbytes = f.read(Plen)

            # Huffman decode the matrix to gain quantized values
            Pq = self.Phuffman.decode_int_matrix(Pbytes, Prows)

            # multiply each value by step to gain original values
            phases = self.Pdequantize_vec(Pq)

            # read and multiply NMF chunks to obtain magnitude matrix
            chunks = list()

            for _ in range(chunk_count):
                # read minimum value
                min_val = struct.unpack('<d', f.read(8))[0]

                # read min and max for re-scaling
                matrix_min, matrix_max = struct.unpack('<dd', f.read(16))

                # read companded scaled matrix W
                Wscs = deserialize_matrix(f, 'I')

                # read Huffman encoded matrix H
                Hrows, Hlen = struct.unpack('<II', f.read(8))
                Hbytes = f.read(Hlen)

                # scale matrix W back
                Wsc = self.scale_matrix(Wscs, 0, 2 ** 32, 0, 1)

                # Huffman decode the matrix to gain quantized values
                Hscq = self.Hhuffman.decode_int_matrix(Hbytes, Hrows)

                # multiply each value by step to gain original values
                Hsc = self.Hdequantize_vec(Hscq)

                # expand the scaled matrices using mu-law
                Ws = self.expand(Wsc, self.MU_LAW_W)
                Hs = self.expand(Hsc, self.MU_LAW_H)

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
