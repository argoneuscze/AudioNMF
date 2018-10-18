from abc import ABC


class AudioFormat(ABC):
    def get_nmf_data(self, input_file, nmf_data):
        raise NotImplementedError

    def write_data(self, input_file, output_file):
        raise NotImplementedError
