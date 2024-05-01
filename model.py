import numpy as np


class ActivationFunctions:
    def __init__(self):
        self.functions = {
            'softmax': (self.softmax, self.softmax_grad),
            'crossentropy': (self.cross_entropy, self.cross_entropy_grad),
            'relu': (self.relu, self.relu_grad),
            'sigmoid': (self.sigmoid, self.sigmoid_grad),
            'tanh': (self.tanh, self.tanh_grad)
        }

    def get_function(self, name):
        if name.lower() in self.functions:
            return self.functions[name.lower()][0]
        else:
            raise ValueError(f"Function '{name}' not recognized.")

    def get_gradient(self, name):
        if name.lower() in self.functions:
            return self.functions[name.lower()][1]
        else:
            raise ValueError(f"Gradient for function '{name}' not recognized.")

    def softmax(self, x):
        s = np.exp(x)
        return s / np.sum(s, axis=1, keepdims=True)

    def softmax_grad(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    def cross_entropy(self, y_pred, y_true):
        epsilon = 1e-12  # to prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    def cross_entropy_grad(self, y_pred, y_true):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return - (y_true / y_pred)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh(self, x):
        return np.tanh(x)
    def tanh_grad(self, x):
        return 1 - np.tanh(x)**2    

class nn_Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Initialize weights randomly
        self.biases = np.zeros((1, output_size))  # Initialize biases to zero
        self.grad_weights = np.zeros_like(self.weights)  # Gradients of weights
        self.grad_biases = np.zeros_like(self.biases)  # Gradients of biases

    def forward(self, input_data):
        self.input_data = input_data  # Store input data for use in backward pass
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output


class Network():
    def __init__(self, size_list, activation_fn_list, activation_fn_grad_list,lr,l2_reg):
        layer1 = nn_Layer(size_list[0], size_list[1])
        layer2 = nn_Layer(size_list[1], size_list[2])
        layer3 = nn_Layer(size_list[2], size_list[3])
        self.layers = [layer1,layer2,layer3]
        self.activation_fns = activation_fn_list
        self.activation_fn_grads = activation_fn_grad_list
        self.lr = lr
        self.l2_reg = l2_reg
        # for i in range(len(size_list) - 1):
        #     self.layers.append(nn_Layer(size_list[i], size_list[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            x = self.activation_fns[i](x)
        return x

    def backward(self, X, y):
        m = X.shape[0]
        dZ3 = self.layers[-1].output - self.one_hot(y, self.layers[-1].output_size)
        dW3 = np.dot(self.layers[-2].output.T, dZ3) / m + self.l2_reg / m * self.layers[-1].weights
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.layers[-1].weights.T)
        dZ2 = dA2 * self.activation_fn_grads[-2](self.layers[-2].output)
        dW2 = np.dot(self.layers[-3].output.T, dZ2) / m + self.l2_reg / m * self.layers[-2].weights
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.layers[-2].weights.T)
        dZ1 = dA1 * self.activation_fn_grads[-3](self.layers[-3].output)
        dW1 = np.dot(X.T, dZ1) / m + self.l2_reg / m * self.layers[-3].weights
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.layers[-3].grad_weights = dW1
        self.layers[-3].grad_biases = db1
        self.layers[-2].grad_weights = dW2
        self.layers[-2].grad_biases = db2
        self.layers[-1].grad_weights = dW3
        self.layers[-1].grad_biases = db3   
        for i in range(len(self.layers)):
            self.layers[i].weights -= self.lr * self.layers[i].grad_weights
            self.layers[i].biases -= self.lr * self.layers[i].grad_biases


    def cross_entropy_loss(self, y_pred, y_true):
        m = y_pred.shape[0]
        loss = -np.sum(np.log(y_pred[np.arange(m), y_true])) / m
        return loss

    def softmax(self, x):
        s = np.exp(x)
        return s / np.sum(s, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        m = y.shape[0]
        one_hot_y = np.zeros((m, num_classes))
        one_hot_y[np.arange(m), y] = 1
        return one_hot_y
    