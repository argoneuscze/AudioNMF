import numpy
from scipy.io import wavfile

from audionmf.formats.audio_format import AudioFormat
from audionmf.nmf.channel import Channel


class AudioFormatWAV(AudioFormat):
    def get_nmf_data(self, wav_file, nmf_data):
        rate, raw_data = wavfile.read(wav_file)
        nmf_data.sample_rate = rate
        for i in range(numpy.size(raw_data, 1)):
            c = Channel()
            c.add_sample_array(raw_data[:, i])
            nmf_data.add_channel(c)

    def write_data(self, nmf_data, output_file):
        wavfile.write(output_file, nmf_data.rate, nmf_data.samples)  # rate and data TODO
