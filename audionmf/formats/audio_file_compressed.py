from audionmf.formats.audio_format_wav import AudioFormatWAV
from audionmf.nmf.nmf_data import NMFData


class AudioFileCompressed:
    def __init__(self, input_file):
        self.input_file = input_file

    def decompress(self, output_file, filetype):
        data = NMFData.from_file(self.input_file)

        audio_format = None
        if filetype == 'wav':
            audio_format = AudioFormatWAV()

        if audio_format is not None:
            data.write_audio_file(output_file, audio_format)

    @staticmethod
    def read_file(handle):
        return AudioFileCompressed(handle)
