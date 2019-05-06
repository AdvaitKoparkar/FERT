from dataloaders.fer_loader import FERDataset
import matplotlib.pyplot as plt
from models.knn import KNNClassifier

if __name__ == '__main__':
    dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='train')
    classifier = KNNClassifier(dset, **{'k': 5})
    classifier.train()
    test_dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='test')
    img, label = test_dset[0]
    y_pred = classifier.test(img)
    print("Predicted emotion: %s, True emotion: %s" %(dset.classes[y_pred[0]], dset.classes[label]))
