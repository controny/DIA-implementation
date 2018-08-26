import numpy as np


class Hyperparams(object):
    """Contains hyper parameters of training."""
    lr = 20
    lr_decay = 0.02
    lr_updated_iters = 50
    momentum = 0.9
    batch_size = 1024
    max_num_epochs = 5
    regularization_rate = 0.0001


class DPPModel(object):
    """Implements DPP-based utilities."""

    def __init__(self, hyperparams):
        """Initializes the model."""
        self.hyperparams = hyperparams
        self.weights = None

    def train(self, features, labels):
        """
        Trains DPP model given training data.
        :param features: (#examples, feature size)
        :param labels: (#examples, #tags)
        """
        cur_iter = 0
        for epoch in range(self.hyperparams.max_num_epochs):
            self.shuffle_data(features, labels)
            for i in range(len(features) / self.hyperparams.batch_size):
                batch_features, batch_labels = self.next_batch(features, labels)
                self.update_weights(batch_features, batch_labels)
                cur_loss = self.compute_loss(batch_features, batch_labels)
                cur_iter += 1
                if self.check_convergence(cur_loss):
                    break

    def shuffle_data(self, features, labels):
        pass

    def next_batch(self, features, labels):
        pass

    def update_weights(self, batch_features, batch_labels):
        pass

    def compute_mean_loss(self, batch_features, batch_labels, similarity_mat):
        """
        Computes mean loss of negative log likehood
        :param batch_features: (batch size, feature size)
        :param batch_labels: (batch size, #tags)
        :param similarity_mat: (#tags, #tags)
        :return mean loss across the batch
        """
        q = self.compute_quality_term(batch_features)  # (batch size, #tags, 1)
        kernels = np.matmul(q, np.transpose(q, [0, 2, 1])) * similarity_mat  # (batch size, #tags, #tags)
        sub_kernels = [kernel[np.ix_(label, label)] for kernel, label in zip(kernels, batch_labels)]
        losses = [self.compute_negative_log_likehood(kernel, sub_kernel)
                  for kernel, sub_kernel in zip(kernels, sub_kernels)]
        return sum(losses) / len(losses)

    def compute_quality_term(self, batch_features):
        """Gets quality term with shape of (batch size, #tag, 1)."""
        q = np.exp(0.5 * np.matmul(batch_features, self.weights))
        q = np.expand_dims(q, -1)
        return q

    @staticmethod
    def compute_negative_log_likehood(kernel, sub_kernel):
        """Gets single loss of negative log likehood."""
        p = np.linalg.det(sub_kernel) / np.linalg.det(kernel + np.identity(len(kernel)))
        return -np.log(p)

    def check_convergence(self, cur_loss):
        pass
