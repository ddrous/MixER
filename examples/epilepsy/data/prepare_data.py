#%%

import sktime
# from sktime.datasets import load_data
import numpy as np
import sktime.datasets
import matplotlib.pyplot as plt

## Open the 'Epilpesy2_TRAIN.ts' dataset
# data = np.loadtxt('data/Epilepsy2_TRAIN.ts', delimiter=',', skiprows=1)

base_filename = 'data/Epilepsy2'
train_X, train_y = sktime.datasets.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
test_X, test_y = sktime.datasets.load_from_tsfile_to_dataframe(str(base_filename) + '_TEST.ts')
train_X = train_X.to_numpy()
test_X = test_X.to_numpy()
# X = np.concatenate((train_X, test_X), axis=0)
# y = np.concatenate((train_y, test_y), axis=0)

#%%
print("Shapes are: ", train_X.shape, test_X.shape, train_y.shape, test_y.shape)

## Plot the training data
plt.plot(train_X[0,0].squeeze())
# plt.plot(train_y.squeeze())
plt.show()

# print(train_X)

#%%

train_X[0,0]

## Put the data in shape (80, 178)
X = []
for i in range(train_X.shape[0]):
    X.append(train_X[i,0].squeeze())

X = np.array(X)

print(X.shape)

## Plot the training data
plt.cla()
plt.plot(X[50])

t = np.linspace(0, 1, X.shape[1])
## Save the dataset to a npz file
np.savez('data/train_data.npz', signal=X, condition=train_y)

#%%
## Do something similar for the test data
X = []
for i in range(test_X.shape[0]):
    X.append(test_X[i,0].squeeze())

X = np.array(X)
print(X.shape)
np.savez('data/adapt_data.npz', signal=X, condition=test_y)