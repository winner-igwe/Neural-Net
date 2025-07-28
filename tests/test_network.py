
import numpy as np
from  nnfs.utils import load_data
from nnfs import network
from nnfs.regularizers import L1, L2


training_data, validation_data = load_data.load_data_wrapper()
net = network.UpdatedNetwork([784,30,10])

epochs = 30
eta = 0.001
lmbda = 0.05
batch_size = 10
evaluation_data=validation_data


evaluation_accuracy, evaluation_cost,training_accuracy, training_cost, stopping_epoch = net.SGD(
    epochs, eta, batch_size, training_data,
            evaluation_data=evaluation_data,
            early_stopping =False,
            lmbda=lmbda,
            delta=0.0005,
            patience=10,
            learning_schedule=False,
            schedule_factor = 0.5,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost = True,
            monitor_training_accuracy=True,
            monitor_training_cost = True,
    )

