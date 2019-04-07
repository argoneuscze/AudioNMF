# average frequencies extracted with get_quant_freq.py
import numpy
from dahuffman import HuffmanCodec

frequencies = {
    'stft32': {
        0: 19820,
        1: 362,
        2: 276,
        3: 277,
        4: 316,
        5: 360,
        6: 347,
        7: 336,
        8: 316,
        9: 295,
        10: 273,
        11: 250,
        12: 264,
        13: 283,
        14: 318,
        15: 334,
        16: 380,
        17: 423,
        18: 384,
        19: 375,
        20: 370,
        21: 352,
        22: 313,
        23: 270,
        24: 218,
        25: 157,
        26: 102,
        27: 61,
        28: 38,
        29: 16,
        30: 6,
        31: 2
    }
}


class HuffmanCoder:
    def __init__(self, method):
        try:
            freqs = frequencies[method]
            self.codec = HuffmanCodec.from_frequencies(freqs)
        except KeyError:
            raise KeyError('Invalid Huffman dictionary.')

    def print_dict(self):
        self.codec.print_code_table()

    def encode_int_array(self, ary):
        return self.codec.encode(ary)

    def decode_int_array(self, raw_bytes):
        return self.codec.decode(raw_bytes)

    def encode_int_matrix(self, matrix):
        rows = matrix.shape[0]
        ary = matrix.reshape(-1).tolist()[0]
        return self.encode_int_array(ary), rows

    def decode_int_matrix(self, raw_bytes, rows):
        ary = self.decode_int_array(raw_bytes)
        return numpy.reshape(ary, (rows, -1))
