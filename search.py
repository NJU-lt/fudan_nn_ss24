from model import Network
from train import train
from model import ActivationFunctions
from data_loader import DataLoader

def grid_search(X_train, y_train, X_val, y_val, input_size, output_size):
    learning_rates = [0.01,0.001,0.0001]
    activations = ActivationFunctions()
    activation_fn_list = [activations.get_function('tanh'), activations.get_function('tanh'), activations.get_function('softmax')]
    activation_fn_grad_list = [activations.get_gradient('tanh'), activations.get_gradient('tanh'), activations.get_gradient('softmax')]
    hidden_sizes_1 = [512,256,128,64,32]
    hidden_sizes_2 = [256,128,64,32,16]

    reg_lambdas = [0.01,0.05,0.1]
    batch_sizes=[32,64]
    best_val_acc = 0
    best_params = {}
    results = []

    for lr in learning_rates:
        for hidden_size_1 in hidden_sizes_1:
            for hidden_size_2 in hidden_sizes_2:
                for reg_lambda in reg_lambdas:
                    for batch_size in batch_sizes:
                        model = Network([784,hidden_size_1,hidden_size_2,10], activation_fn_list, activation_fn_grad_list,lr,reg_lambda)
                        print(f"Training with lr={lr}, hidden_size_1={hidden_size_1}, hidden_size_2={hidden_size_2}, reg_lambda={reg_lambda}, batch_size={batch_size}")
                        _,val_acc = train(model, X_train, y_train, X_val, y_val, 30, batch_size,if_plot=False)            
                        results.append((lr, hidden_size_1, hidden_size_2, reg_lambda, batch_size, val_acc))
                        print(f"Validation accuracy: {val_acc:.4f}")
                        
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = {'lr': lr, 'hidden_size_1': hidden_size_1, 'hidden_size_2': hidden_size_2, 'reg_lambda': reg_lambda, 'batch_size': batch_size}
                            print(f"New best params found: {best_params}, val_acc: {val_acc:.4f}")

    # 保存或返回结果
    return best_params, best_val_acc, results

dataloader = DataLoader('./data')
X_train, y_train, X_val, y_val, X_test, y_test = dataloader.load_data()

best_params, best_accuracy,result=grid_search(X_train,y_train,X_val, y_val, input_size=784, output_size=10)
print("result:",result)
print("Best params:", best_params)
print("Best validation accuracy:", best_accuracy)