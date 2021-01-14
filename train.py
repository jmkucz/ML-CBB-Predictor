"""
Modified model from University of Michigan EECS 445 material
"""
import torch
import numpy as np
import random
from model import NN
import utils
from combine import get_data

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predictions(logits):
    """
    Given the network output, determines the predicted class index

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    # TODO: implement this function
    pred = torch.argmax(logits, dim=1)
    return pred
    #

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion):
    """
    Evaluates the `model` on the train and validation set.
    """
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in tr_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
    train_loss = np.mean(running_loss)
    train_acc = correct / total
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in val_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
    val_loss = np.mean(running_loss)
    val_acc = correct / total
    print("val_acc: " + str(val_acc))
    print("val_loss: " + str(val_loss))
    print("train_acc: " + str(train_acc))
    print("train_loss: " + str(train_loss))

def main():
    # Data loaders
    tr_loader, va_loader = get_data()

    # Model
    model = NN()

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    #

    print('Number of float-valued parameters:', count_parameters(model))

    _train_epoch(tr_loader, model, criterion, optimizer)

    # Evaluate model
    _evaluate_epoch(tr_loader, va_loader, model, criterion)



    print('Finished Training')


if __name__ == '__main__':
    main()
