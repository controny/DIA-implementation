import numpy as np
from utils import timeit


class Hyperparams(object):
    """Contains hyper parameters of training."""
    def __init__(self, feat_size, num_tags):
        self.feat_size = feat_size
        self.num_tags = num_tags
        self.initial_lr = 20
        self.lr_decay = 0.02
        self.lr_updated_iters = 50
        self.momentum = 0.9
        self.batch_size = 1024
        self.max_num_epochs = 5
        self.regularization_rate = 0.0001


class DPPModel(object):
    """Implements DPP-based utilities."""

    def __init__(self, hyperparams):
        """Initializes the model."""
        self.hyperparams = hyperparams
        self.weights = np.random.normal(size=(self.hyperparams.feat_size, self.hyperparams.num_tags))
        self.momentum_grad = np.zeros(self.weights.shape)
        self.lr = hyperparams.initial_lr

    @timeit
    def train(self, features, labels, similarity_mat):
        """
        Trains DPP model given training data.
        :param features: (#examples, feature size)
        :param labels: (#examples, #tags)
        :param similarity_mat: (#tags, #tags)
        """
        cur_iter = 0
        for epoch in range(self.hyperparams.max_num_epochs):
            print('=' * 10 + ' Epoch %d ' % epoch + '=' * 10)
            self.shuffle_data(features, labels)
            iters_per_epoch = int(len(features) / self.hyperparams.batch_size)
            for i in range(iters_per_epoch):
                cur_loss = self.single_train_iteration(features, labels, similarity_mat, cur_iter, i)
                cur_iter += 1
                if self.check_convergence(cur_loss):
                    break

    @timeit
    def single_train_iteration(self, features, labels, similarity_mat, global_iter, local_iter):
        """
        Train a single iteration.
        :param features: (#examples, feature size)
        :param labels: (#examples, #tags)
        :param similarity_mat: (#tags, #tags)
        :param global_iter: current global steps
        :param local_iter: current local steps within an epoch
        :return current loss
        """
        batch_features, batch_labels = self.next_batch(features, labels, local_iter)
        kernels = self.compute_kernels(batch_features, similarity_mat)
        raw_grad = self.compute_raw_gradient(batch_features, batch_labels, kernels)
        cur_loss = self.compute_mean_loss(batch_labels, kernels)
        self.gradient_check(raw_grad, batch_features, batch_labels, similarity_mat)
        self.update_weights(raw_grad)
        self.update_learning_rate(global_iter)
        print('Iterations %d: loss = %f' % (global_iter, cur_loss))

        return cur_loss

    @staticmethod
    def shuffle_data(features, labels):
        """
        Shuffle features and labels simultaneously.
        :param features: (#examples, feature size)
        :param labels: (#examples, #tags)
        """
        zipped = list(zip(features, labels))
        np.random.shuffle(zipped)
        features[:], labels[:] = zip(*zipped)

    def next_batch(self, features, labels, i):
        """
        Get the next training batch.
        :param features: (#examples, feature size)
        :param labels: (#examples, #tags)
        :param i: current training step within an epoch
        :return a tuple of batch data
        """
        start = i * self.hyperparams.batch_size
        end = start + self.hyperparams.batch_size
        batch_features = features[start:end]
        batch_labels = labels[start:end]
        return batch_features, batch_labels

    @staticmethod
    def compute_raw_gradient(batch_features, batch_labels, kernels):
        """
        Compute raw mean gradient across a batch without regularization term.
        :param batch_features: (batch size, feature size)
        :param batch_labels: (batch size, #tags)
        :param kernels: (batch size, #tags, #tags)
        :return mean gradient
        """
        grads = []
        for i in range(len(batch_features)):
            u, s, _ = np.linalg.svd(kernels[i])
            # note that `s` is a vector instead of matrix here
            squared_u = u ** 2  # column vectors are vi^2
            transformed_s = s / (s + 1)  # elements are `\lambda_{i} / (\lambda_{i} + 1)`
            temp_kiis = squared_u * transformed_s  # should be shape of (#tags, #tags) due to broadcast rule
            kiis = np.sum(temp_kiis, axis=1)  # with shape of (#tags, )
            # should convert vectors to matrixes
            cur_grad = np.matmul(batch_features[i][:, np.newaxis], (kiis - batch_labels[i])[:, np.newaxis].T)
            grads.append(cur_grad)
        return np.mean(grads, axis=0)

    @timeit
    def update_weights(self, raw_grad):
        """
        Update weights using SGD.
        :param raw_grad: raw gradient without regularization term.
        """
        gradient = raw_grad + self.hyperparams.regularization_rate * self.weights
        self.momentum_grad = self.hyperparams.momentum * self.momentum_grad - self.lr * gradient
        self.weights += self.momentum_grad

    @timeit
    def compute_mean_loss(self, batch_labels, kernels):
        """
        Computes mean loss of negative log likehood
        :param batch_labels: (batch size, #tags)
        :param kernels: (batch size, #tags, #tags)
        :return mean loss across the batch
        """
        # extract corresponding indexes where the element is 1
        batch_label_indexes = [np.nonzero(label)[0] for label in batch_labels]
        losses = []
        for i in range(len(kernels)):
            kernel = kernels[i]
            label_indexes = batch_label_indexes[i]
            sub_kernel = kernel[np.ix_(label_indexes, label_indexes)]
            loss = self.compute_negative_log_likehood(kernel, sub_kernel)
            losses.append(loss)
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

    def update_learning_rate(self, cur_iter):
        pass

    def gradient_check(self, grad, batch_features, batch_labels, similarity_mat):
        """Check if gradient is computed correctily."""
        # define an internal function to compute loss by change an element of weights temporarily
        def get_loss(model, indexes, diff):
            model.weights[indexes] += diff
            kernels = model.compute_kernels(batch_features, similarity_mat)
            loss = model.compute_mean_loss(batch_labels, kernels)
            model.weights[indexes] -= diff
            return loss

        # Iterate over all indexes iw in weights to check the gradient.
        it = np.nditer(self.weights, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            iw = it.multi_index
            delta = 0.0001
            forward = get_loss(self, iw, delta)
            backward = get_loss(self, iw, -delta)
            numerical_grad = (forward - backward) / (2 * delta)
            # Compare gradients
            # note that `numerical_grad` is a scalar
            reldiff = abs(numerical_grad - grad[iw]) / max(1, abs(numerical_grad), abs(grad[iw]))
            if reldiff > 1e-5:
                print("First gradient error found at index %s" % str(iw))
                print('My gradient = %f but numerical gradient = %f'
                      % (grad, numerical_grad))
                return
            it.iternext()
        print('Gradient check passed!')
