from glob import glob
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split


dataset = 'camUPDRS'  # 'HAR', camUPDRS
data_dir = './data/' + dataset #r"np"
seed = 3
p = 25   # percentage of data to use

if dataset == 'camUPDRS':
    files = sorted(glob(os.path.join(data_dir, "*.npz")))
    train_full_dict = np.load(data_dir + '/train.npz')
    test_full_dict = np.load(data_dir + '/test.npz')
    test_dict = {'samples': test_full_dict['samples'], 'labels': test_full_dict['labels']}

    # split train into train and val
    split = 0.2
    train_idxs = np.arange(len(train_full_dict['samples']))
    np.random.shuffle(train_idxs)
    val_idxs = train_idxs[:int(split * len(train_idxs))]
    train_idxs = train_idxs[int(split * len(train_idxs)):]

    # make sure val set has desired num of 0 labels
    NUM_VAL_0s = 6
    if len(val_idxs) - train_full_dict['labels'][val_idxs].sum() < NUM_VAL_0s:
        # train_idxs = np.arange(len(train_full_dict['samples']))
        # np.random.shuffle(train_idxs)
        # val_idxs = train_idxs[:int(split * len(train_idxs))]
        # train_idxs = train_idxs[int(split * len(train_idxs)):]
        
        # add more 0 label data to val set, remove from train set
        train_0s_idxs = np.where(train_full_dict['labels'] == 0)[0]
        np.random.shuffle(train_0s_idxs)
        num_val_0s_curr = len(val_idxs) - train_full_dict['labels'][val_idxs].sum()
        num_val_0s_diff = NUM_VAL_0s - num_val_0s_curr
        val_idxs = np.append(val_idxs, train_0s_idxs[:num_val_0s_diff])
        train_idxs = np.setdiff1d(train_idxs, train_0s_idxs[:num_val_0s_diff])
        

    train_dict = {'samples': train_full_dict['samples'][train_idxs], 'labels': train_full_dict['labels'][train_idxs]}
    val_dict = {'samples': train_full_dict['samples'][val_idxs], 'labels': train_full_dict['labels'][val_idxs]}

    # convert to tensors
    train_dict['samples'] = torch.from_numpy(train_dict['samples']).permute(0, 2, 1)
    train_dict['labels'] = torch.from_numpy(train_dict['labels'])

    val_dict['samples'] = torch.from_numpy(val_dict['samples']).permute(0, 2, 1)
    val_dict['labels'] = torch.from_numpy(val_dict['labels'])

    test_dict['samples'] = torch.from_numpy(test_dict['samples']).permute(0, 2, 1)
    test_dict['labels'] = torch.from_numpy(test_dict['labels'])

    # resave main files as .pt
    torch.save(train_dict, data_dir + '/train.pt')
    torch.save(val_dict, data_dir + '/val.pt')    
    torch.save(test_dict, data_dir + '/test.pt')


files = sorted(glob(os.path.join(data_dir, "*.pt")))
train_full_dict = torch.load(data_dir + '/train.pt')
val_dict = torch.load(data_dir + '/val.pt')

# Train
# split into 99% and 1%
x_train_full = train_full_dict['samples']
y_train_full = train_full_dict['labels']

# include all 0's:
X_train_nonfrac, X_train_frac, y_train_nonfrac, y_train_frac = train_test_split(x_train_full, y_train_full, test_size=p/100, random_state=seed)

# move 0's from y_train_nonfrac to y_train_frac
nonfrac_0_idxs = np.where(y_train_nonfrac == 0)[0]
y_train_frac = np.append(y_train_frac, y_train_nonfrac[nonfrac_0_idxs])
X_train_frac = np.append(X_train_frac, X_train_nonfrac[nonfrac_0_idxs], axis=0)

train_frac_dict = {}
train_frac_dict['samples'] = X_train_frac
train_frac_dict['labels'] = y_train_frac

# Val
x_val = val_dict['samples']
y_val = val_dict['labels']

# ensure we have enough label == 0:
X_val_nonfrac, X_val_frac, y_val_nonfrac, y_val_frac = train_test_split(x_val, y_val, test_size=p/100, random_state=seed)
# while y_val_frac.shape[0] - y_val_frac.sum().item() >= 0.6 * NUM_VAL_0s:
#     _, X_val_frac, _, y_val_frac = train_test_split(x_val, y_val, test_size=p/100, random_state=seed+1)
# move 0's from y_val_nonfrac to y_val_frac
nonfrac_0_idxs = np.where(y_val_nonfrac == 0)[0]
y_val_frac = np.append(y_val_frac, y_val_nonfrac[nonfrac_0_idxs])
X_val_frac = np.append(X_val_frac, X_val_nonfrac[nonfrac_0_idxs], axis=0)

val_frac_dict = {}
val_frac_dict['samples'] = X_val_frac
val_frac_dict['labels'] = y_val_frac

torch.save(train_frac_dict, data_dir + '/train_1perc.pt')
torch.save(val_frac_dict, data_dir + '/val_1perc.pt')

# print stats of new files
print("train_frac_dict: ", train_frac_dict['samples'].shape)
print(" no. 0s/1s: ", (train_frac_dict['labels'] == 0).sum().item(), (train_frac_dict['labels'] == 1).sum().item())
print("val_frac_dict: ", val_frac_dict['samples'].shape)
print(" no. 0s/1s: ", (val_frac_dict['labels'] == 0).sum().item(), (val_frac_dict['labels'] == 1).sum().item())
print("test_dict: ", test_dict['samples'].shape)
print(" no. 0s/1s: ", (test_dict['labels'] == 0).sum().item(), (test_dict['labels'] == 1).sum().item())

