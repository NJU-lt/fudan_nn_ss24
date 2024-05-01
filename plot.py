
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_acc(train_losses, val_losses, val_accuracies, model, batch_size):
    # 绘制损失曲线和准确率曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'lr:{model.lr},Batch size:{batch_size},l2_reg:{model.l2_reg}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'lr:{model.lr},Batch size:{batch_size},l2_reg:{model.l2_reg}')
    plt.show()

def visualize_layer_parameters(layer, layer_num):
    weights = layer.weights
    biases = layer.biases

    plt.figure(figsize=(12, 6))

    # Plotting weights
    plt.subplot(1, 2, 1)
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"Layer {layer_num} Weights")
    plt.xlabel("Output Neurons")
    plt.ylabel("Input Neurons")

    # Plotting biases
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(biases.shape[1]), biases.flatten())
    plt.title(f"Layer {layer_num} Biases")
    plt.xlabel("Output Neurons")
    plt.ylabel("Bias Value")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"layer_{layer_num}.png")

def visualize_network(model):
    for i, layer in enumerate(model.layers):
        visualize_layer_parameters(layer, i + 1)

