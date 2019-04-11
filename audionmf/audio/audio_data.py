from audionmf.fileformats.audio_format_wav import AudioFormatWAV
from audionmf.nmfcompression.nmfcompressor_mdct import NMFCompressorMDCT
from audionmf.nmfcompression.nmfcompressor_raw import NMFCompressorRaw
from audionmf.nmfcompression.nmfcompressor_stft import NMFCompressorSTFT

audio_formats = {
    'wav': AudioFormatWAV
}

compression_schemes = {
    'anmfs': NMFCompressorSTFT,
    'anmfr': NMFCompressorRaw,
    'anmfm': NMFCompressorMDCT
}


def get_audio_format(string):
    try:
        return audio_formats[string]()
    except KeyError:
        print('Invalid audio format: {}.'.format(string))
        exit(1)


def get_compression_format(string):
    try:
        return compression_schemes[string]()
    except KeyError:
        print('Invalid compression scheme: {}.'.format(string))
        exit(2)


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
