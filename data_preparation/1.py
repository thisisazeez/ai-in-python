import numpy as np
from sklearn import preprocessing

Input_data = np.array([[2.1, -1.9, 5.5],
 [-1.5, 2.4, 3.5],
 [0.5, -7.9, 5.6],
 [5.9, 2.3, -5.8]])


# Binarized Data

data_binarized = preprocessing.Binarizer(threshold=0.5).transform(Input_data)

print("\nBinarized data:\n", data_binarized)

#Mean Removal
data_scaled = preprocessing.scale(Input_data)
print("Mean = ", data_scaled.mean(axis=0))
print("Std Deviation= ", data_scaled.std(axis=0))

#Scaling

data_scalar_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scalar_minmax.fit_transform(Input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

#Normalize data
#L1 normalization
data_normalized_l1 = preprocessing.normalize(Input_data, norm='l1')
print("\nL1 normalized data:\n", data_normalized_l1)

#L2 normalization
data_normalized_l2 = preprocessing.normalize(Input_data, norm='l2')
print("\nL2 normalized data:\n", data_normalized_l2)