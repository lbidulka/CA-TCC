import torch
import numpy as np
import os
import copy


# TRAIN AND TEST A SIMPLE MLP

config = {
    # Model parameters
    "data_channels": 4,
    "input_dim": 128,
    "hidden_dim": 128,
    "output_dim": 1,
    "dropout": 0.30,

    # Training parameters
    "batch_size": 8,
    "num_epochs": 25,
    "learning_rate": 5e-4,

    # Data parameters
    "shuffle": True,
    "drop_last": True,

    # Experiment parameters
    "frac_data": False   # only use small fraction of data?
}

# Seed for reproducibility
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

# define model
class MLP(torch.nn.Module):
    def __init__(self, data_channels, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.input_dim = input_dim * data_channels
        self.fc1 = torch.nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# load dataset from .pt files
data_path = 'data/camUPDRS'
train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt" if config["frac_data"] else "train.pt"))
valid_dataset = torch.load(os.path.join(data_path, "val_1perc.pt" if config["frac_data"] else "val.pt"))
test_dataset = torch.load(os.path.join(data_path, "test.pt"))

# convert to torch tensors
X_train = torch.tensor(train_dataset["samples"], dtype=torch.float32)
y_train = torch.tensor(train_dataset["labels"], dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(valid_dataset["samples"], dtype=torch.float32)
y_val = torch.tensor(valid_dataset["labels"], dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(test_dataset["samples"], dtype=torch.float32)
y_test = torch.tensor(test_dataset["labels"], dtype=torch.float32).unsqueeze(1)

# Print data info
print("\n|| Data info: ||")
print("X_train: ", X_train.shape)
print("X_val: ", X_val.shape)
print("X_test: ", X_test.shape)
print()

# make tensor datasets
trainset = torch.utils.data.TensorDataset(X_train, y_train)
valset = torch.utils.data.TensorDataset(X_val, y_val)
testet = torch.utils.data.TensorDataset(X_test, y_test)

# dataloaders
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config["batch_size"],
                                           shuffle=config["shuffle"], drop_last=config['drop_last'], num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=config["batch_size"],
                                            shuffle=False, drop_last=config['drop_last'], num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testet, batch_size=config["batch_size"],
                                            shuffle=False, drop_last=config['drop_last'], num_workers=0)

# initialize model
model = MLP(config["data_channels"], config["input_dim"], config["hidden_dim"], config["output_dim"], config["dropout"])

# define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# train model
print("\nTraining model...")
best_val_loss = np.inf
for epoch in range(config["num_epochs"]):
    for i, (x, y) in enumerate(train_loader):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get acc
        predicted = (y_pred > 0.5) * 1.0
        total = y.size(0)
        correct = (predicted == y).sum().item()
        # update acc
        if i == 0:
            acc = correct / total
        else:
            acc = 0.5 * acc + 0.5 * (correct / total)

    print("Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, config["num_epochs"], 
                                                                loss.item(), acc))
    
    # evaluate model
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = []
        for x, y in val_loader:
            y_pred = model(x.float())
            val_loss.append(criterion(y_pred, y))
            predicted = (y_pred > 0.5) * 1.0
            total += y.size(0)
            correct += (predicted == y).sum().item()
        val_loss = torch.mean(torch.tensor(val_loss))
        print('Val Loss: {:.4f},  Acc: {:.4f}'.format(val_loss, correct / total), end='')
        if val_loss < best_val_loss:
            print("  --> best model!", end='')
            best_val_loss = val_loss
            # best_model = copy.deepcopy(model)
            ckpt_dir = './ckpts_baselines/'
            torch.save(model.state_dict(), ckpt_dir + 'best_model.pt')
        print()
    
# test model
model.load_state_dict(torch.load(ckpt_dir + 'best_model.pt'))
with torch.no_grad():
    correct = 0
    total = 0
    for x, y in test_loader:
        y_pred = model(x.float())
        predicted = (y_pred > 0.5) * 1.0
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print('\n|| Test Acc: {:.4f} ||\n'.format(correct / total, 4))
            
