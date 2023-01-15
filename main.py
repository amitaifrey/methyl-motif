import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold


K = 30


class MethylCNN(nn.Module):
    def __init__(self):
        super(MethylCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=6, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (K - 4 + 1) * (K - 4 + 1), 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (K - 4 + 1) * (K - 4 + 1))
        x = F.relu(self.fc1(x))
        return x


ENC = {"A": 0, "C": 1, "G": 2, "T": 3}


def hot_encode(seq: str):
    res = np.zeros((4,len(seq)))
    for i, c in enumerate(list(seq)):
        res[ENC[c]][i] = 1

    return res.T


def main():
    model = MethylCNN()
    print(model)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # K-fold cross validation
    k = 5  # number of folds
    num_epochs = 10

    # Split the data into k-folds
    kf = KFold(n_splits=k, shuffle=True)

    plus_file = open('plus_seq2.txt', 'r')
    pluses = plus_file.readlines()
    X_pluses = np.array([hot_encode(p.strip()) for p in pluses])
    y_pluses = np.ones(len(pluses))

    minus_file = open('plus_seq2.txt', 'r')
    minuses = minus_file.readlines()
    X_minuses = np.array([hot_encode(m.strip()) for m in minuses])
    y_minuses = np.zeros(len(minuses))

    X = np.r_[X_pluses, X_minuses]
    y = np.r_[y_pluses, y_minuses]

    for train_index, val_index in kf.split(X):
        # Split the data into train and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert data to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        X_val = torch.from_numpy(X_val).float()
        y_train = torch.from_numpy(y_train).long()
        y_val = torch.from_numpy(y_val).long()

        # Train the model
        for epoch in range(num_epochs):
            # Forward pass
            output = model(X_train)
            loss = criterion(output, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set
        with torch.no_grad():
            output = model(X_val)
            val_loss = criterion(output, y_val)
            val_acc = (output.argmax(1) == y_val).float().mean()

        # Print the validation loss and accuracy
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')


if __name__ == "__main__":
    main()
