from abc import ABC


class AudioFormat(ABC):
    def get_nmf_data(self, input_file, nmf_data):
        raise NotImplementedError

    def write_file(self, nmf_data, output_file):
        raise NotImplementedError
