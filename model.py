import numpy as np
import random
import time
import matplotlib.pyplot as plt
from utils import timeit


class Hyperparams(object):
    """Contains hyper parameters of training."""
    def __init__(self, feat_size, num_tags):
        self.feat_size = feat_size
        self.num_tags = num_tags
        self.initial_lr = 50
        self.lr_decay = 0.02
        self.lr_updated_iters = 50
        self.momentum = 0.9
        self.batch_size = 1024
        self.max_num_epochs = 5
        self.max_iters = 200
        self.regularization_rate = 0.0001


class DPPModel(object):
    """Implements DPP-based utilities."""

    def __init__(self, hyperparams):
        """Initializes the model."""
        self.hyperparams = hyperparams
        self.weights = np.random.normal(size=(self.hyperparams.feat_size, self.hyperparams.num_tags))
        self.momentum_grad = np.zeros(self.weights.shape)
        self.lr = hyperparams.initial_lr
        self.losses = []

    @timeit
    def train(self, features, labels, similarity_mat):
        """
        Trains DPP model given training data.
        :param features: (#examples, feature size)
        :param labels: (#examples, #tags)
        :param similarity_mat: (#tags, #tags)
        """
        cur_iter = 0
        stop = False
        for epoch in range(self.hyperparams.max_num_epochs):
            print('=' * 10 + ' Epoch %d ' % epoch + '=' * 10)
            self.shuffle_data(features, labels)
            iters_per_epoch = int(len(features) / self.hyperparams.batch_size)
            for i in range(iters_per_epoch):
                cur_loss = self.single_train_iteration(features, labels, similarity_mat, cur_iter, i)
                self.losses.append(cur_loss)
                cur_iter += 1
                if cur_iter >= self.hyperparams.max_iters or self.check_convergence(cur_loss):
                    stop = True
                    break
            if stop:
                break

        self.plot_losses(cur_iter, 'data/result.png')

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
        # self.gradient_check(raw_grad, batch_features, batch_labels, similarity_mat)
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

    def update_weights(self, raw_grad):
        """
        Update weights using SGD.
        :param raw_grad: raw gradient without regularization term.
        """
        gradient = raw_grad + self.hyperparams.regularization_rate * self.weights
        self.momentum_grad = self.hyperparams.momentum * self.momentum_grad - self.lr * gradient
        self.weights += self.momentum_grad

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

    def compute_quality_terms(self, batch_features):
        """
        Computes quality terms for a batch
        :param batch_features: (batch size, feature size)
        :return quality terms with shape of (batch size, #tags)
        """
        return np.exp(0.5 * np.matmul(batch_features, self.weights))

    def compute_kernels(self, batch_features, similarity_mat):
        """
        Computes kernels for a batch
        :param batch_features: (batch size, feature size)
        :param similarity_mat: (#tags, #tags)
        :return (batch size, #tags, #tags)
        """
        q = self.compute_quality_terms(batch_features)
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
            # sample check
            if random.random() > 0.0001:
                it.iternext()
                continue
            iw = it.multi_index
            print("Checking gradient at index %s" % str(iw))
            delta = 0.0001
            forward = get_loss(self, iw, delta)
            backward = get_loss(self, iw, -delta)
            numerical_grad = (forward - backward) / (2 * delta)
            # Compare gradients
            # note that `numerical_grad` is a scalar
            reldiff = abs(numerical_grad - grad[iw]) / max(1, abs(numerical_grad), abs(grad[iw]))
            if reldiff > 1e-5:
                print('My gradient = %f but numerical gradient = %f'
                      % (grad, numerical_grad))
                return
            it.iternext()
        print('Gradient check passed!')

    def plot_losses(self, iters, filename):
        """Plot the losses and save to an image file."""
        plt.plot(range(iters), self.losses)
        plt.xscale('log')
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.savefig(filename)
        print('Result image saved at %s' % filename)

    def inference(self, features, semantic_paths, similarity_mat, k1=8, k2=5, num_trials=10):
        """
        Inference on multiple features.
        :param features: (#examples, feature size)
        :param similarity_mat: (#tags, #tags)
        :param k1: size of the reduced set at first step
        :param k2: maximum size of the final result
        :param num_trials: number of trials to take
        :return: inference results with shape of (#examples, #tags)
        """
        kernels = self.compute_kernels(features, similarity_mat)
        q = self.compute_quality_terms(features)
        # TODO: should sort in descending order and partition
        k1_tags_indexes = np.argpartition(q, -k1, axis=1)[:, range(-k1, 0)]  # (#instances, k1)
        results = np.zeros([features.shape[0], similarity_mat.shape[0]])
        for i in range(len(kernels)):
            kernel = kernels[i]
            indexes = k1_tags_indexes[i]
            sub_kernel = kernel[np.ix_(indexes, indexes)]
            max_weights_sum = 0
            max_trail = None
            for t in range(num_trials):
                trial = self.k_dpp_sample(semantic_paths, k1_tags_indexes[i], sub_kernel, k2)
                weights_sum = self.get_SP_weights_sum(trial, semantic_paths)
                if weights_sum > max_weights_sum:
                    max_weights_sum = weights_sum
                    max_trail = trial
            results[i, :] = max_trail
        return results

    def k_dpp_sample(self, semantic_paths, candidate_tags, kernel, k):
        """
        Perform k-dpp sampling for single instance input.
        """
        u, lmbdas, _ = np.linalg.svd(kernel)
        elem_sympolys = self.compute_elementary_symmetric_polynomials(lmbdas, k)
        # The first phase: select a set of vectors to construct an elementary DPP
        vector_indexes = []
        counter = k
        for n in range(len(candidate_tags)):
            # here we don't exclude some tags by SP, which will be done by controlling probabilities below
            rand = random.uniform(0.0, 1.0)
            if rand < lmbdas[n] * elem_sympolys[k-1][n-1] / elem_sympolys[k][n]:
                vector_indexes.append(candidate_tags[n])
                counter -= 1
                if counter == 0:
                    break
        V = u[:, vector_indexes]  # (#candidate tags size, k)
        # The second phase: sample the answer according to the elementary DPP mentioned above
        answer = np.zeros([len(candidate_tags)])
        for _ in range(k):
            # compute probabilities for each item
            probs = np.sum(V**2, axis=1)
            # TODO: set the probabilities of elements in the existing paths as 0
            probs = probs / np.sum(probs)
            i = np.random.choices(candidate_tags, p=probs)
            answer[i] = 1
            V = self.get_subspace(V, i)
            self.orthonormalize(V)
        return answer

    def get_SP_weights_sum(self, tags, semantic_paths):
        pass

    @staticmethod
    def get_subspace(V, i):
        """Gets subspace of V orthogonal to e_{i}."""
        j = np.nonzero(V[i, :])[0][0]
        Vj = V[:, j]
        # remove Vj
        V = np.delete(V, j, axis=1)
        # colomn transform
        V -= np.outer(Vj, V[i, :] / Vj[i])
        return V

    @staticmethod
    def orthonormalize(V):
        """Orthonormalize V using Gram-Schmidt process."""
        num_cols = V.shape[1]
        for a in range(0, num_cols):
            for b in range(0, a):
                V[:, a] = V[:, a] - np.dot(V[:, a], V[:, b]) * V[:, b]
            norm = np.linalg.norm(V[:, a])
            V[:, a] = V[:, a] / norm

    @staticmethod
    def compute_elementary_symmetric_polynomials(lmbdas, k):
        """
        Compute elementary symmetric polynomials recursively
        and return them as a 2-D array.
        :param lmbdas eigenvalues
        :param k size of set
        :return a 2-D array, where `e[l][n]` denotes e_{l}^{n}
        """
        N = len(lmbdas)
        e = np.zeros([k+1, N+1])
        # initialize base situations
        e[0, :] = 1
        for l in range(1, k+1):
            for n in range(1, N+1):
                # note that `lmbdas` has regular index starting from 0, different from the formula
                e[l][n] = e[l][n-1] + lmbdas[n-1] * e[l-1][n-1]
        return e
