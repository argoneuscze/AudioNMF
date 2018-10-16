from scipy.io import wavfile

from audionmf.formats.audio_format import AudioFormat
from audionmf.nmf.nmf_data import NMFData


class AudioFormatWAV(AudioFormat):
    def get_nmf_data(self, wav_file):
        rate, raw_data = wavfile.read(wav_file)
        data = NMFData()

        # TODO fill data

        return data

    def write_data(self, data, output_file):
        wavfile.write(output_file, data.rate, data.samples)  # rate and data TODO
