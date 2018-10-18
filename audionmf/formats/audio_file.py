from audionmf.formats.audio_format_wav import AudioFormatWAV
from audionmf.nmf.nmf_data import NMFData


class AudioFile:
    def __init__(self, input_file, audio_format):
        self.input_file = input_file
        self.audio_format = audio_format

    def compress(self, output_file):
        print('compressing {} to {} using WAV'.format(self.input_file, output_file))
        data = NMFData()
        self.audio_format.get_nmf_data(self.input_file, data)
        data.write_file(output_file)

    @staticmethod
    def read_file(handle, filetype):
        if filetype == 'wav':
            return AudioFile(handle, AudioFormatWAV())
        return None
