from dataloaders.fer_loader import FERDataset
import matplotlib.pyplot as plt
from models.svm import SVMClassifier
from utils.f1_score import F1_score

if __name__ == '__main__':
    dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='train')
    classifier = SVMClassifier(dset, **{'C': 1})
    classifier.train()
    test_dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='test')
    images = []
    labels = []
    y_pred = []
    for i in range(0,len(test_dset)):
        img, label = test_dset[i]
        y_pred.append(classifier.test(img))
        images.append(img)
        labels.append(label)
    f1_pred = F1_score(**{'filename':'svm_f1.png', 'y_true':labels, 'y_pred': y_pred, 'model_name': 'svm'})
    f1_pred.metric_plot()
    #print("Predicted emotion: %s, True emotion: %s" %(dset.classes[y_pred[0]], dset.classes[label]))
