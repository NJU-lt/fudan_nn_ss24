import numpy as np
import pickle
from data_loader import DataLoader
from model import Network
from model import ActivationFunctions
from plot import plot_loss_and_acc

def evaluate(y_pred, y):
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred == y)
    return accuracy

def train(model,X_train, y_train, X_val, y_val, num_epochs=50, batch_size=32,if_plot=True):
    best_val_accuracy = 0
    best_weight = None
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_train_loss = []

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            y_pred = model.forward(X_batch)
            loss = model.cross_entropy_loss(y_pred, y_batch)
            m = len(y_batch)
            loss += 0.5/m * model.l2_reg * (np.sum(model.layers[0].weights ** 2) + np.sum(model.layers[1].weights ** 2)+ np.sum(model.layers[2].weights ** 2))
            epoch_train_loss.append(loss)
            model.backward(X_batch, y_batch)

        y_pred = model.forward(X_val)
        val_loss = model.cross_entropy_loss(y_pred, y_val)
        val_accuracy = evaluate(y_pred, y_val)

        train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {round(val_accuracy,3)}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weight = {
                'W1': model.layers[0].weights,
                'b1': model.layers[0].biases,
                'W2': model.layers[1].weights,
                'b2': model.layers[1].biases,
                'W3': model.layers[2].weights,
                'b3': model.layers[2].biases
            }
    if if_plot:
        plot_loss_and_acc(train_losses, val_losses, val_accuracies, model, batch_size)

    return best_weight, best_val_accuracy


if __name__ == '__main__':
    dataloader = DataLoader('./data')
    X_train, y_train, X_val, y_val, X_test, y_test = dataloader.load_data()
    num_epochs, batch_size = 30, 32

    activations = ActivationFunctions()
    activation_fn_list = [activations.get_function('tanh'), activations.get_function('tanh'), activations.get_function('softmax')]
    activation_fn_grad_list = [activations.get_gradient('tanh'), activations.get_gradient('tanh'), activations.get_gradient('softmax')]

    model = Network([784,256,128,10], activation_fn_list, activation_fn_grad_list,0.01, 0.01)
    model_weight, val_accuracy = train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size)
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file, True)



    

    

    

