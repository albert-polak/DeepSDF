import numpy as np

data1 = np.load('/home/albert/experiment_8instances/data/SdfSamples/shapeNetDataset/bottle/1ef68777bfdb7d6ba7a07ee616e34cd7/pos.npy')
data2 = np.load('/home/albert/experiment_8instances/data/SdfSamples/shapeNetDataset/bottle/1ef68777bfdb7d6ba7a07ee616e34cd7/neg.npy')

combined_data = np.concatenate((data1, data2), axis=0)

np.save('combined.npy', combined_data)
