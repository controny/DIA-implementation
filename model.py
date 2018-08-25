import numpy


class Hyperparams(object):
    """Contains hyper parameters of training."""
    pass


class DPPModel(object):
    """Implements DPP-based utilities."""

    def __init__(self, hyperparams, word_embeds):
        """Initializes the model."""
        self.word_embeds = word_embeds
        self.num_tags = None
        self.weights = None

    def train(self):
        pass
