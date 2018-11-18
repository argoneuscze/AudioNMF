from abc import ABC


class AudioFormat(ABC):
    def fill_audio_data(self, input_fd, audio_data):
        raise NotImplementedError

    def write_file(self, audio_data, output_fd):
        raise NotImplementedError
