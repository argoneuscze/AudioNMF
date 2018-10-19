class NMFData:
    def __init__(self):
        self.sample_rate = None
        self.channels = list()
        ...  # TODO various metadata, rate, samples, etc.

    def add_channel(self, channel):
        self.channels.append(channel)

    def write_nmf_file(self, output_file):
        ...  # TODO NMF compress into .anmf

    def write_audio_file(self, output_file, audio_format):
        audio_format.write_file(self, output_file)

    @staticmethod
    def from_file(input_file):
        ...  # TODO NMF decompress from .anmf and return it
