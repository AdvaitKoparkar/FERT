from utils.tsne import TSNE
from dataloaders.fer_loader import FERDataset
from models.tinyresnet import TinyResNet

if __name__ == "__main__":
    tsne_creator = TSNE(**{'dset': FERDataset, 'model': TinyResNet})
    tsne_creator.reduce()
    tsne_creator.visualize()
