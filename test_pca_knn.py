from dataloaders.fer_loader import FERDataset
import matplotlib.pyplot as plt
from models.knn import KNNClassifier
from experiments.classification import Classification
from sklearn.decomposition import PCA
import numpy as np
import pdb 

#Testing PCA with KNNClassifier
#Check test_pca_knn2.py for KNNCLassifier_withPCA
if __name__ == '__main__':
    dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='train')
#    classification_knn = Classification(**{'classifier': KNNClassifier, 'dset':FERDataset, 'max_samples': 1000})
#    pdb.set_trace()
    
    
    classifier = KNNClassifier(dset, **{'k': 5})
    classifier._load_data()
    pca = PCA(n_components=200)
    pca.fit(classifier.X)
    dset_pca=pca.transform(classifier.X)
    
    
    test_dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='test')
    classifier=KNNClassifier(test_dset, **{'k': 5})
    classifier._load_data()
    test_pca=pca.transform(classifier.X)
    
    classifier.X=dset_pca
    classifier.train()
    y_pred=classifier.test(test_pca)
    
    label=classifier.y[0][0]
    
    print("Predicted emotion: %s, True emotion: %s" %(dset.classes[y_pred[0]], dset.classes[label]))

