import numpy as np
import struct
from pathlib import Path
import random

# Get the current file's directory (nnfs/utils/)
current_dir = Path(__file__).parent

# Point to data directory (nnfs/data/mnist/)
data_dir = current_dir.parent / "data" / "mnist"

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num, rows, cols))
        return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_data_wrapper():
    images = load_images(data_dir / "train-images.idx3-ubyte")
    labels = load_labels(data_dir / "train-labels.idx1-ubyte")
    images = np.reshape(images, (60000, 784, 1))
    
    combined_data = [(x,y) for (x,y) in zip(images,labels)]
    random.shuffle(combined_data)
    
    training_temp, eval_temp = combined_data[:50000], combined_data[50000:]
    for index in range(len(training_temp)):
        hold = np.zeros((10,1)) 
        x,y = training_temp[index]
        hold[y] = 1
        training_temp[index] = x,hold
         
    training_data, eval_data = training_temp, eval_temp
    return training_data, eval_data
training_data, eval_data = load_data_wrapper()
# print(len(training_data))
