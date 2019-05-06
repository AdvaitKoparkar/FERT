from experiments.classification import Classification
from dataloaders.fer_loader import FERDataset
from models.knn_PCA import KNNClassifier_withPCA
from models.logreg import LogRegClassifier
from models.svm import SVMClassifier

if __name__ == '__main__':

#     classification_knn = Classification(**{'classifier': KNNClassifier, 'dset':FERDataset})
#     classification_knn.grid_search(hyper_params={'k':[5], 'weights': ['distance']})
#    classification_svm = Classification(**{'classifier': SVMClassifier, 'dset':FERDataset, 'save_path': 'runs/my_svm.pkl'})
#    classification_svm.grid_search(hyper_params={'C':[0.1], 'kernel': ['rbf'], 'decision_function_shape':['ovr']})
#    classification_lr = Classification(**{'classifier': LogRegClassifier, 'dset':FERDataset})
#    classification_lr.grid_search(hyper_params={'multi_class':['multinomial','ovr'], 'solver':['newton-cg'], 'C':[1,1e5,1e-2]})
     classification_knn_pca = Classification(**{'classifier': KNNClassifier_withPCA, 'dset':FERDataset})
     classification_knn_pca.grid_search(hyper_params={'k':[5, 10], 'weights': ['distance'], 'pca_ncomps':[200, 500, 750]})

     classification_knn_pca.gen_metrics()
#    classification_svm.gen_metrics()
#    classification_lr.gen_metrics()

    # classification_lr.visualize_best_model()
