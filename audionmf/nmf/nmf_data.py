class NMFData:
    def __init__(self):
        ...  # TODO various metadata, rate, samples, etc.

    def write_file(self, output_file):
        ...  # TODO NMF compress into .anmf

    @staticmethod
    def from_file(input_file):
        ...  # TODO NMF decompress from .anmf
