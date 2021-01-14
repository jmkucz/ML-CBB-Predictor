"""
Torch Attempt
"""
from combine import get_data
from model import NN

if __name__ == "__main__":
    train_loader, val_loader = get_data()

    model = NN()

