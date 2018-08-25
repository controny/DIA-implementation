import os
import scipy.io
import numpy as np


class Dataset(object):
    """
    Provides the following data:
    1. PCA-processed VGG feature vectors
    2. S matrix computed based on GloVe
    3. Semantic paths
    """

    def __init__(self, data_name):
        self.data_name = data_name
        self.data_dir = os.path.join('./data', data_name)

    def load_feat_vectors(self, is_training=True):
        """Loads feature vectors as nparray with shape of (#examples, feature size)."""
        file_name = '%s_data_vggf_pca_%s.txt' % (self.data_name, 'train' if is_training else 'test')
        res = np.loadtxt(os.path.join(self.data_dir, file_name), delimiter=',')
        return res

    def load_similarity_matrix(self):
        """Loads S matrix as nparray with shape of (#tags, #tags)."""
        file_name = 'S_psd_gloVe_%s.mat' % self.data_name
        mat = scipy.io.loadmat(os.path.join(self.data_dir, file_name))
        return mat['S']
