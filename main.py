import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import altair as alt
from sklearn.metrics import r2_score

K = 30
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 50
POLES = 1_000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# basic CNN model
class DNA_CNN(nn.Module):
    def __init__(self, seq_len, num_filters=32, kernel_size=10, device=DEVICE):
        super().__init__()
        self.seq_len = seq_len

        self.conv_net = nn.Sequential(
            # 4 is for the 4 nucleotides
            nn.Conv1d(4, num_filters, kernel_size=kernel_size, padding=1, device=device),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(num_filters, 64, kernel_size=5, padding=1, device=device),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64*10, 64, device=device),
            nn.Linear(64, 16, device=device),
            nn.Linear(16, 1, device=device),
        )

    def forward(self, xb):
        # permute to put channel in correct order
        # (batch_size x 4channel x seq_len)
        xb = xb.permute(0, 2, 1)

        # print(xb.shape)
        out = self.conv_net(xb)
        return out


ENC = {"A": 0, "C": 1, "G": 2, "T": 3}


def hot_encode(seq: str):
    res = np.zeros((4, len(seq)))
    for i, c in enumerate(list(seq)):
        res[ENC[c]][i] = 1

    return res.T


def loss_batch(model, loss_func, xb, yb, opt=None, verbose=False):
    '''
    Apply loss function to a batch of inputs. If no optimizer
    is provided, skip the back prop step.
    '''
    if verbose:
        print('loss batch ****')
        print("xb shape:", xb.shape)
        print("yb shape:", yb.shape)
        print("yb shape:", yb.squeeze(1).shape)
        # print("yb",yb)

    # get the batch output from the model given your input batch
    # ** This is the model's prediction for the y labels! **
    xb_out = model(xb.float())

    if verbose:
        print("model out pre loss", xb_out.shape)
        # print('xb_out', xb_out)
        print("xb_out:", xb_out.shape)
        print("yb:", yb.shape)
        print("yb.long:", yb.long().shape)

    loss = loss_func(xb_out, yb.float())  # for MSE/regression

    if opt is not None:  # if opt
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def train_step(model, train_dl, loss_func, device, opt):
    '''
    Execute 1 set of batched training within an epoch
    '''
    # Set model to Training mode
    model.train()
    tl = []  # train losses
    ns = []  # batch sizes, n

    # loop through train DataLoader
    for xb, yb in train_dl:
        # put on GPU
        xb, yb = xb.to(device), yb.to(device)

        # provide opt so backprop happens
        t, n = loss_batch(model, loss_func, xb, yb, opt=opt)

        # collect train loss and batch sizes
        tl.append(t)
        ns.append(n)

    # average the losses over all batches
    train_loss = np.sum(np.multiply(tl, ns)) / np.sum(ns)

    return train_loss


def val_step(model, val_dl, loss_func, device):
    '''
    Execute 1 set of batched validation within an epoch
    '''
    # Set model to Evaluation mode
    model.eval()
    with torch.no_grad():
        vl = []  # val losses
        ns = []  # batch sizes, n

        # loop through validation DataLoader
        for xb, yb in val_dl:
            # put on GPU
            xb, yb = xb.to(device), yb.to(device)

            # Do NOT provide opt here, so backprop does not happen
            v, n = loss_batch(model, loss_func, xb, yb)

            # collect val loss and batch sizes
            vl.append(v)
            ns.append(n)

    # average the losses over all batches
    val_loss = np.sum(np.multiply(vl, ns)) / np.sum(ns)

    return val_loss


def fit(epochs, model, loss_func, opt, train_dl, val_dl, device,
        patience=1000):
    '''
    Fit the model params to the training data, eval on unseen data.
    Loop for a number of epochs and keep train of train and val losses
    along the way
    '''
    # keep track of losses
    train_losses = []
    val_losses = []

    # loop through epochs
    for epoch in range(epochs):
        # take a training step
        train_loss = train_step(model, train_dl, loss_func, device, opt)
        train_losses.append(train_loss)

        # take a validation step
        val_loss = val_step(model, val_dl, loss_func, device)
        val_losses.append(val_loss)

        print(f"E{epoch} | train loss: {train_loss:.3f} | val loss: {val_loss:.3f}")

    return train_losses, val_losses


def run_model(train_dl, val_dl, model, device='cpu', lr=LR, epochs=EPOCHS, lossf=None, opt=None):
    '''
    Given train and val DataLoaders and a NN model, fit the mode to the training
    data. By default, use MSE loss and an SGD optimizer
    '''
    # define optimizer
    if opt:
        optimizer = opt
    else:  # if no opt provided, just use SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define loss function
    if lossf:
        loss_func = lossf
    else:  # if no loss function provided, just use MSE
        loss_func = torch.nn.MSELoss()

    # run the training loop
    train_losses, val_losses = fit(epochs, model, loss_func, optimizer, train_dl, val_dl, device)

    return train_losses, val_losses


def quick_split(df, split_frac=0.8):
    '''
    Given a df of samples, randomly split indices between
    train and test at the desired fraction
    '''
    cols = df.columns  # original columns, use to clean up reindexed cols
    df = df.reset_index()

    # shuffle indices
    idxs = list(range(df.shape[0]))
    random.shuffle(idxs)

    # split shuffled index list by split_frac
    split = int(len(idxs) * split_frac)
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]

    # split dfs and return
    train_df = df[df.index.isin(train_idxs)]
    test_df = df[df.index.isin(test_idxs)]

    return train_df[cols], test_df[cols]


class SeqDatasetOHE(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''

    def __init__(self, df, seq_col='seq', target_col='score'):
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])
        self.ohe_seqs = torch.stack([torch.tensor(hot_encode(x)) for x in self.seqs])
        self.labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.ohe_seqs[idx]
        label = self.labels[idx]

        return seq, label


def build_dataloaders(train_df, test_df, seq_col='data', target_col='label',
                      batch_size=BATCH_SIZE, shuffle=True):
    '''
    Given a train and test df with some batch construction
    details, put them into custom SeqDatasetOHE() objects.
    Give the Datasets to the DataLoaders and return.
    '''

    # create Datasets
    train_ds = SeqDatasetOHE(train_df, seq_col=seq_col, target_col=target_col)
    test_ds = SeqDatasetOHE(test_df, seq_col=seq_col, target_col=target_col)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl


def quick_loss_plot(data_label_list, loss_type="MSE Loss", sparse_n=0):
    '''
    For each train/test loss trajectory, plot loss by epoch
    '''
    for i, (train_data, test_data, label) in enumerate(data_label_list):
        plt.plot(train_data, linestyle='--', color=f"C{i}", label=f"{label} Train")
        plt.plot(test_data, color=f"C{i}", label=f"{label} Val", linewidth=3.0)

    plt.legend()
    plt.ylabel(loss_type)
    plt.xlabel("Epoch")
    plt.legend(bbox_to_anchor=(1, 1), loc='lower right')
    plt.show()


def parity_plot(model_name, df, r2):
    '''
    Given a dataframe of samples with their true and predicted values,
    make a scatterplot.
    '''
    plt.scatter(df['truth'].values, df['pred'].values, alpha=0.2)

    # y=x line
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False,scaley=False)

    plt.ylim(xpoints)
    plt.ylabel("Predicted Score", fontsize=14)
    plt.xlabel("Actual Score", fontsize=14)
    plt.title(f"{model_name} Poles: {POLES}", fontsize=20)
    plt.show()


def alt_parity_plot(model, df, r2):
    '''
    Make an interactive parity plot with altair
    '''
    chart = alt.Chart(df).mark_circle(size=100, opacity=0.4).encode(
        alt.X('truth:Q'),
        alt.Y('pred:Q'),
        tooltip=['seq:N']
    ).properties(
        title=f'{model} (r2:{r2:.3f})'
    ).interactive()

    chart.save(f'alt_out/parity_plot_{model}.html')
    display(chart)


def parity_pred(models, seqs, oracle, alt=False):
    '''Given some sequences, get the model's predictions '''
    dfs = {}  # key: model name, value: parity_df

    for model_name, model in models:
        print(f"Running {model_name}")
        data = []
        for dna in seqs:
            s = torch.tensor(hot_encode(dna)).unsqueeze(0).to(DEVICE)
            actual = oracle[dna]
            pred = model(s.float())
            data.append([dna, actual, pred.item()])
        df = pd.DataFrame(data, columns=['seq', 'truth', 'pred'])
        r2 = r2_score(df['truth'], df['pred'])
        dfs[model_name] = (r2, df)

        # plot parity plot
        if alt:  # make an altair plot
            alt_parity_plot(model_name, df, r2)
        else:
            parity_plot(model_name, df, r2)


def main():
    df_plus = pd.read_csv('plus_20_seq10k.txt', header=None)
    df_plus.columns = ["data", "label"]
    df_minus = pd.read_csv('minus_20_seq10k.txt', header=None)
    df_minus.columns = ["data", "label"]
    # data is sorted by %, asc
    df_plus, df_minus = df_plus.tail(POLES), df_minus.head(POLES)

    df = pd.concat([df_plus, df_minus], axis=0).reset_index(drop=True)
    full_train_df, test_df = quick_split(df)
    train_df, val_df = quick_split(full_train_df)

    train_dl, val_dl = build_dataloaders(train_df, val_df)

    model = DNA_CNN(K, 32)
    train_losses, val_losses = run_model(train_dl, val_dl, model, DEVICE)
    data_label = (train_losses, val_losses, "Loss")
    quick_loss_plot([data_label])

    seqs = test_df['data'].values
    models = [("CNN", model)]
    oracle = df.set_index('data').to_dict()['label']
    parity_pred(models, seqs, oracle)


if __name__ == "__main__":
    main()
