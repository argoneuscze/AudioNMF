from audionmf.fileformats.audio_format_wav import AudioFormatWAV
from audionmf.nmfcompression.nmfcompressor import NMFCompressor


def get_audio_format(string):
    if string == 'wav':
        return AudioFormatWAV()
    raise KeyError('Invalid audio format.')


def get_compression_format(string):
    if string == 'anmf':
        return NMFCompressor()
    raise KeyError('Invalid compression format.')


class AudioData:
    def __init__(self):
        self.sample_rate = 0  # todo sanity check
        self.channels = list()
        ...  # TODO various metadata, rate, samples, etc.

    def add_channel(self, channel):
        self.channels.append(channel)

    def write_audio_file(self, output_fd, audio_format_str):
        audio_format = get_audio_format(audio_format_str)
        audio_format.write_file(self, output_fd)

    def write_compressed_file(self, output_fd, compressor_str):
        compressor = get_compression_format(compressor_str)
        compressor.compress(self, output_fd)

    @staticmethod
    def from_audio_file(input_fd, filetype):
        audio_format = get_audio_format(filetype)
        data = AudioData()
        audio_format.fill_audio_data(input_fd, data)
        return data

    @staticmethod
    def from_compressed_file(input_fd, filetype):
        decompressor = get_compression_format(filetype)
        data = AudioData()
        decompressor.decompress(input_fd, data)
        return data
