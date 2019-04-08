# average frequencies extracted with get_quant_freq.py
import numpy
from dahuffman import HuffmanCodec

frequencies = {
    'stft32': {
        0: 2522,
        1: 374,
        2: 280,
        3: 287,
        4: 313,
        5: 321,
        6: 319,
        7: 306,
        8: 301,
        9: 277,
        10: 263,
        11: 251,
        12: 260,
        13: 293,
        14: 315,
        15: 350,
        16: 378,
        17: 412,
        18: 390,
        19: 388,
        20: 374,
        21: 369,
        22: 330,
        23: 281,
        24: 227,
        25: 163,
        26: 106,
        27: 66,
        28: 39,
        29: 0,
        30: 8,
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
        ary = matrix.reshape(-1).tolist()
        return self.encode_int_array(ary), rows

    def decode_int_matrix(self, raw_bytes, rows):
        ary = self.decode_int_array(raw_bytes)
        return numpy.reshape(ary, (rows, -1))
