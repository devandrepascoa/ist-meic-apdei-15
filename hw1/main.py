# This is a sample Python script.
import numpy as np

from utils import load_oct_data

if __name__ == '__main__':
    dataset = load_oct_data()
    print(np.unique(dataset['train'][1]))
    print(dataset['train'][0].shape)