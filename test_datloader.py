from dataloaders.fer_loader import FERDataset
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dset = FERDataset(root='./data/fer2013', filename='fer2013.csv', mode='train', tensor=True)
    dataloader = DataLoader(dset, batch_size=4, shuffle=True, num_workers=1)
    for idx, data in enumerate(dataloader):
        img, label = data
        # img_np = img.numpy() if you want to convert to numpy
        grid_img = torchvision.utils.make_grid(img, nrow=int(img.shape[0]/2))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
