import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pylab
from utils.data import *
from utils.mlp_utils import *

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        layers = []
        in_features = input_feature_count
        for out_features, activation in zip(neuron_counts, activations):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation)
            in_features = out_features
        layers.pop()  # remove last activation for output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

if __name__ == "__main__":

    # Configuration from mlp.yml
    input_feature_count = 9
    neuron_counts = [64, 64, 1]
    activations = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()]
    loss_func = nn.MSELoss(reduction='mean')
    learn_rate = 0.01
    lambda_L2 = 0.001
    batch_size = 128
    epochs = 500
    device = torch.device("cpu")
    save_dir = "."

    # Dummy dataset (replace with actual data as needed)
    train_size = 500
    X = torch.randn(train_size, input_feature_count)
    y = torch.randn(train_size, 1)
    X, y = gen_matrix_stack(train_size, int(input_feature_count**(1/2)))

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=lambda_L2)

    # Training loop
    epoch_plt = []
    loss_plt = []

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_func(output, batch_y)
            loss.backward()
            optimizer.step()

        epoch_plt.append(epoch + 1)
        loss_plt.append(loss.item())

        print(f"epoch = {epoch+1}, loss = {loss.item()}")
        print(f"__________________________________________")

    # Plot results
    plot_training_results(epoch_plt, loss_plt, save_dir)


    # Inference (evaluation) step
    model.eval()  # set model to evaluation mode

    # Dummy test data — replace with actual test data as needed
    test_size = 100    
    X_test, y_test = gen_matrix_stack(test_size, int(input_feature_count**(1/2)))

    with torch.no_grad():
        predictions = model(X_test)

    # Optionally evaluate performance
    print_classification_results(test_size, predictions.squeeze(), y_test)
