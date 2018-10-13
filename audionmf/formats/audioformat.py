from abc import ABC


class AudioFormat(ABC):
    def get_samples(self):
        raise NotImplementedError
