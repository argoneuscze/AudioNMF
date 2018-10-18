class NMFData:
    def __init__(self):
        self.sample_rate = 0
        self.channels = list()
        ...  # TODO various metadata, rate, samples, etc.

    def add_channel(self, channel):
        self.channels.append(channel)

    def write_file(self, output_file):
        ...  # TODO NMF compress into .anmf

    @staticmethod
    def from_file(input_file):
        ...  # TODO NMF decompress from .anmf
