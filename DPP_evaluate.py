from dataset import Dataset
from model import Hyperparams, DPPModel


def main():
    data_name = 'espgame'
    data_loader = Dataset(data_name)
    test_features = data_loader.load_feat_vectors(is_training=False)
    test_labels = data_loader.load_semantic_hierachy_labels(is_training=False)
    similarity_mat = data_loader.load_similarity_matrix()
    semantic_data = data_loader.load_semantic_data()
    hyperparams = Hyperparams(test_features.shape[-1], test_labels.shape[-1])
    dpp_model = DPPModel(hyperparams)
    dpp_model.inference(test_features, semantic_data, similarity_mat)


if __name__ == '__main__':
    main()
