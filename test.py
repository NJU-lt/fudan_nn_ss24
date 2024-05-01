import numpy as np
import pickle
from data_loader import DataLoader
from model import Network
from model import ActivationFunctions
from plot import plot_loss_and_acc, visualize_network

def evaluate(y_pred, y):
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred == y)
    return accuracy

def test(model, X_test, y_test):
    y_pred = model.forward(X_test)
    accuracy = evaluate(y_pred, y_test)
    print("Test accuracy:", accuracy)


if __name__ == '__main__':
    dataloader = DataLoader('./data')
    X_train, y_train, X_val, y_val, X_test, y_test = dataloader.load_data()
    num_epochs, batch_size = 30, 32

    activations = ActivationFunctions()
    activation_fn_list = [activations.get_function('tanh'), activations.get_function('tanh'), activations.get_function('softmax')]
    activation_fn_grad_list = [activations.get_gradient('tanh'), activations.get_gradient('tanh'), activations.get_gradient('softmax')]

    with open("model.pkl", "rb") as file:
        best_model = pickle.load(file)
    
    test(best_model, X_test, y_test)

    visualize_network(best_model)

    

    

    

