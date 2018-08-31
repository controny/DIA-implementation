from dataset import Dataset
from model import Hyperparams, DPPModel


def main():
    data_name = 'espgame'
    data_loader = Dataset(data_name)
    train_features = data_loader.load_feat_vectors()
    train_labels = data_loader.load_semantic_hierachy_labels()
    similarity_mat = data_loader.load_similarity_matrix()
    hyperparams = Hyperparams(train_features.shape[-1], train_labels.shape[-1])
    dpp_model = DPPModel(hyperparams)
    dpp_model.train(train_features, train_labels, similarity_mat)


if __name__ == '__main__':
    main()
