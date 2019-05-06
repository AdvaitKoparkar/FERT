from dataloaders.fer_loader import FERDataset
import matplotlib.pyplot as plt
from models.simple_cnn import SimpleCNN

if __name__ == '__main__':
    dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='test', tensor=True)
    classifier = SimpleCNN(dset)
    classifier.train()
