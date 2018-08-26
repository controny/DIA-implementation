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

    def train(self, features, labels, similarity_mat):
        """
        Trains DPP model given training data.
        :param features: (#examples, feature size)
        :param labels: (#examples, #tags)
        :param similarity_mat: (#tags, #tags)
        """
        self.hyperparams.feat_size = features.shape[-1]
        self.hyperparams.num_tags = labels.shape[-1]
        self.weights = np.random.rand(self.hyperparams.feat_size, self.hyperparams.num_tags)
        cur_iter = 0
        for epoch in range(self.hyperparams.max_num_epochs):
            self.shuffle_data(features, labels)
            for i in range(len(features) / self.hyperparams.batch_size):
                batch_features, batch_labels = self.next_batch(features, labels)
                kernels = self.compute_kernels(batch_features, similarity_mat)
                self.update_weights(batch_features, batch_labels)
                cur_loss = self.compute_mean_loss(batch_labels, kernels)
                cur_iter += 1
                if self.check_convergence(cur_loss):
                    break

    def shuffle_data(self, features, labels):
        pass

    def next_batch(self, features, labels):
        pass

    def update_weights(self, batch_features, batch_labels, kernels):
        pass

    def compute_mean_loss(self, batch_labels, kernels):
        """
        Computes mean loss of negative log likehood
        :param batch_labels: (batch size, #tags)
        :param kernels: (batch size, #tags, #tags)
        :return mean loss across the batch
        """
        batch_label_indexes = [np.nonzero(label) for label in batch_labels]
        sub_kernels = [kernel[np.ix_(label_indexes, label_indexes)]
                       for kernel, label_indexes in zip(kernels, batch_label_indexes)]
        losses = [self.compute_negative_log_likehood(kernel, sub_kernel)
                  for kernel, sub_kernel in zip(kernels, sub_kernels)]
        return sum(losses) / len(losses)

    def compute_kernels(self, batch_features, similarity_mat):
        """
        Computes kernels for a batch
        :param batch_features: (batch size, feature size)
        :param similarity_mat: (#tags, #tags)
        :return (batch size, #tags, #tags)
        """
        q = np.exp(0.5 * np.matmul(batch_features, self.weights))
        q = np.expand_dims(q, -1)  # (batch size, #tags, 1)
        kernels = np.matmul(q, np.transpose(q, [0, 2, 1])) * similarity_mat
        return kernels

    @staticmethod
    def compute_negative_log_likehood(kernel, sub_kernel):
        """Gets single loss of negative log likehood."""
        p = np.linalg.det(sub_kernel) / np.linalg.det(kernel + np.identity(len(kernel)))
        return -np.log(p)

    def check_convergence(self, cur_loss):
        pass
