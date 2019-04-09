# average frequencies extracted with get_quant_freq.py
import numpy
from dahuffman import HuffmanCodec

frequencies = {
    'stft32': {
        0: 13404,
        1: 951,
        2: 625,
        3: 503,
        4: 400,
        5: 362,
        6: 316,
        7: 305,
        8: 311,
        9: 327,
        10: 377,
        11: 404,
        12: 400,
        13: 418,
        14: 442,
        15: 452,
        16: 434,
        17: 417,
        18: 396,
        19: 375,
        20: 322,
        21: 281,
        22: 246,
        23: 200,
        24: 138,
        25: 93,
        26: 63,
        27: 43,
        28: 31,
        29: 26,
        30: 14,
        31: 4
    },
    'stftp': {
        0: 25703,
        1: 50394,
        2: 50421,
        3: 50716,
        4: 51780,
        5: 50437,
        6: 50291,
        7: 26557
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
        ary = matrix.reshape(-1).tolist()
        return self.encode_int_array(ary), rows

    def decode_int_matrix(self, raw_bytes, rows):
        ary = self.decode_int_array(raw_bytes)
        return numpy.reshape(ary, (rows, -1))
