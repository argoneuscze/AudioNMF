import numpy
from scipy.io import wavfile

from audionmf.audio.channel import Channel
from audionmf.fileformats.audio_format import AudioFormat


class AudioFormatWAV(AudioFormat):
    def fill_audio_data(self, wav_file_fd, audio_data):
        rate, raw_data = wavfile.read(wav_file_fd)
        if raw_data.dtype != 'int16':
            raise Exception('WAV format must be 16-bit integers')

        # replace all 0s with 1s to prevent division by zero during processing
        raw_data_c = numpy.copy(raw_data)
        raw_data_c[raw_data_c == 0] = 1

        audio_data.sample_rate = rate
        for i in range(numpy.size(raw_data_c, 1)):
            c = Channel()
            c.add_sample_array(raw_data_c[:, i])
            audio_data.add_channel(c)

    def write_file(self, audio_data, output_fd):
        all_samples = numpy.column_stack([samples for samples in (channel.samples for channel in audio_data.channels)])
        wavfile.write(output_fd, audio_data.sample_rate, all_samples)
