from utils.olivetti import OlivettiDataset
import pdb

if __name__ == "__main__":
    olivetti = OlivettiDataset()
    olivetti.generate_dataset()
    olivetti.generate_results()
