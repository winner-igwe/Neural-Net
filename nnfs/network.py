import numpy as np
import copy
import json
import sys
from .costs import CrossEntropyCost
from .activations import Sigmoid, Softmax

#Main Network class
class UpdatedNetwork():
    def __init__(self,layers, cost=CrossEntropyCost, regularization=None):
        self.layers = layers
        self.num_layers = len(layers)
        self.cost = cost
        self.regularization = regularization
        self.default_weight_initializer()
        self.v_w, self.v_b = [np.zeros((a.shape)) for a in self.weights],[np.zeros((a.shape)) for a in self.biases]



    def default_weight_initializer(self):
        self.biases = [np.random.randn(r,1) for r in self.layers[1:]]
        self.weights = [np.random.randn(r,c)/np.sqrt(c) for (r,c) in zip(self.layers[1:], self.layers[:-1])]

        
    def large_weight_initializer(self):
        self.biases = [np.random.randn(r,1) for r in self.layers[1:]]
        self.weights = [np.random.randn(r,c) for (r,c) in zip(self.layers[1:], self.layers[:-1])]



    def feed_forward(self,x):
        a = x
        curr_l =0
        for w,b in zip(self.weights, self.biases):
            z  = np.dot(w,a) + b
            a = Sigmoid.fn(z)
            curr_l += 1
        return a



    def SGD(self, epochs, eta, mini_batch_size, training_data,
            evaluation_data=None,
            early_stopping =False,
            lmbda =0,
            delta=0.0005,
            patience=10,
            learning_schedule=False,
            schedule_factor = 0.5,
            monitor_evaluation_accuracy=False,
            monitor_evaluation_cost = False,
            monitor_training_accuracy=False,
            monitor_training_cost = False,
            ):
        """
        implements a momentum based stochastic gradient descent

        Args:
            epochs- number of training epochs
            eta- learning rate. Can be optionally modified with a learning schedule
            mini_batch_size- size of the mini batch
            training_data- numpy array of shape (m,1). Each row is a tuple of (x,y)
            evaluation_data- Data used for evaluation for each epoch. Defaulted to None.
            early_stopping- Can be set to true if it is required that the network stops, if it doesn't learn past a certain threshold(defined by delta)
            lmbda- Regularisation param, defaulted to 0
            delta- The minimum increase in accuracy on the evaluation data to achieve an "improvement". Default, 0.0005.
            patience- the maximum consistent epochs without improvement for early stopping
            learning_schedule- option to continually decrease the learning rate by a factor (defined by schedule_factor) after every patience threshold
            Flags to monitor accuracy and cost on both the training and evaluation data

        Return:
            Accuracy and Cost perfomance across each epoch if the corresponding flags are set to true

            
        """
        
        train_size = len(training_data)
        if evaluation_data: eval_size = len(evaluation_data)

        evaluation_accuracy, evaluation_cost = [],[]
        training_accuracy, training_cost = [],[]

        stopping_epoch = epochs
        best_acc = 0
        epochs_without_improvement = 0

        eta_sched = 1
        eta = eta * eta_sched
    
        for epoch in range(epochs):
        
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,train_size,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_weights(mini_batch,eta,lmbda)

            print("Finished training epoch %s" %epoch)

            if monitor_training_cost:
                t_cost = self.total_cost(training_data,lmbda)
                training_cost.append(t_cost)
                print("cost in training data: %s" %t_cost)

            if monitor_evaluation_cost:
                e_cost = self.total_cost(evaluation_data,lmbda, convert=True)
                evaluation_cost.append(e_cost)
                print("cost in evaluation data: %s" %e_cost)

            if monitor_training_accuracy:
                t_accuracy = self.accuracy(training_data,convert=True)
                t_acc_rat = t_accuracy/train_size
                training_accuracy.append(t_acc_rat)
                print("accuracy on training data: {}".format(t_acc_rat))

            if monitor_evaluation_accuracy:
                e_accuracy = self.accuracy(evaluation_data)
                e_acc_rat = e_accuracy/eval_size
                evaluation_accuracy.append(e_acc_rat)
                print("accuracy on evaluation data: {}".format(e_acc_rat))

            if early_stopping:
                if evaluation_accuracy[-1] > best_acc + delta:
                    epochs_without_improvement=0
                    best_acc = evaluation_accuracy[-1]
                    best_weights = copy.deepcopy(self.weights)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    if eta_sched > (schedule_factor)**5 and learning_schedule:
                        eta_sched *= schedule_factor
                        epochs_without_improvement = 0
                        print("decreasing learning rate by half... \n")
                    else:
                        print("Early stopping at epoch: {}!".format(epoch))
                        stopping_epoch = epoch + 1
                        break
                    self.weights = best_weights

                


        return evaluation_accuracy, evaluation_cost,training_accuracy, training_cost, stopping_epoch


    def update_weights(self, mini_batch, eta, lmbda,):
        mu=0.9
        nabla_w, nabla_b = self.backpropagate(mini_batch)
        m = len(mini_batch)

        for i in range(len(self.weights)):
            self.v_w[i] = mu* self.v_w[i] - (eta/m)*nabla_w[i]
            self.v_b[i] = mu* self.v_b[i] - eta*nabla_b[i]

            if self.regularization:
                self.weights[i] = self.weights[i] - (lmbda/m)*self.regularization.reg_weights(lmbda, self.weights[i])
                
            self.weights[i] = self.weights[i] + self.v_w[i]
            self.biases[i] = self.biases[i] + self.v_b[i]


  
    def backpropagate(self,mini_batch):

        nabla_w = [np.zeros((w.shape)) for w in self.weights]
        nabla_b = [np.zeros((b.shape)) for  b in self.biases]
        X = np.concatenate((tuple([x for (x,_) in mini_batch])),axis=1)
        Y = np.concatenate((tuple([y for (_,y) in mini_batch])),axis=1)


        activation = X
        activations=[X]
        Zs = []
        curr_l = 0
        

        for W,b in zip(self.weights, self.biases):
            Z = W @ activation + b
            activation = Sigmoid.fn(Z)
            curr_l +=1
            activations.append(activation)
            Zs.append(Z)

        delta = self.cost.delta(activations[-1],Y,Zs[-1])
        nabla_w[-1] = delta @ activations[-2].T
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)

        for l in range(2,self.num_layers):
            z = Zs[-l]
            sp = Sigmoid.fn_prime(z)
            delta = (self.weights[-l+1].T @ delta) * sp
            nabla_w[-l] = delta @ activations[-l-1].T
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)


        return nabla_w, nabla_b
    
    def total_cost(self,data, lmbda, convert=False):
        cost = 0
        for (x,y) in data:
            if convert: y = self.vectorize(y) 
            a = self.feed_forward(x)
            cost += self.cost.compute_cost(a,y)/len(data)

        cost += self.regularization.cost(lmbda,len(data),self.weights) if self.regularization else 0
        return cost
    

    def accuracy(self,data, convert= False):
        results = [(np.argmax(self.feed_forward(x)),np.argmax(y)) for x,y in data] if convert else [(np.argmax(self.feed_forward(x)),y) for x,y in data]
        return sum(int(x==y) for x,y in results)
    

    def vectorize(self,y):
        v = np.zeros((self.layers[-1],1))
        v[y] = 1
        return v
    
    

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"layers": self.layers,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__),
                "regularization":str(self.regularization.__name__)
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    


def load(filename):
    """Load a neural network from the file ``filename``. Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    reularization= getattr(sys.modules[__name__],data["regularization"])
    cost = getattr(sys.modules[__name__], data["cost"])
    net = UpdatedNetwork(data["sizes"], cost=cost, regularization=reularization)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net




            
