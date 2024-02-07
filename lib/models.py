import torch
import torch.nn as nn
import math

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split


class BaseLearnerRegression(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        activation,
        loss_function,
        dropout_rates=None,
        lr=0.001,
        annealing=None,
        ncl_lambda=None,
    ):
        super(BaseLearnerRegression, self).__init__()

        # Check if the number of layers, activations, and dropout rates match
        if dropout_rates != None:
            assert len(hidden_sizes) == len(
                dropout_rates
            ), "Mismatch in parameter lengths"

        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(self.get_activation(activation))
        if dropout_rates != None:
            layers.append(nn.Dropout(p=dropout_rates[0]))

        # Add hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(self.get_activation(activation))
            if dropout_rates != None:
                layers.append(nn.Dropout(p=dropout_rates[i]))

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))

        # Define the sequential model
        self.input_size = input_size
        self.output_size = 1
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.model = nn.Sequential(*layers)
        self.loss_function = loss_function
        self.lr = lr
        self.annealing = annealing
        self.ncl_lambda = ncl_lambda
        self.loss_curve = []

    def forward(self, x):
        return self.model(x).squeeze()

    def get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        else:
            raise NotImplementedError(f"{activation} activation not implemented")

    def setup_data_loaders(self, df_train_set, validation_size=None):
        self.train_loader = None
        self.val_loader = None

        if validation_size != None:
            train_data, val_data = train_test_split(
                df_train_set, test_size=0.2, random_state=42
            )
        else:
            train_data = df_train_set

        self.train_data_x = torch.tensor(
            train_data.drop(columns=["target"]).values, dtype=torch.float32
        )
        self.train_data_y = torch.tensor(
            train_data["target"].values, dtype=torch.float32
        )

        if validation_size != None:
            self.val_data_x = torch.tensor(
                val_data.drop(columns=["target"]).values, dtype=torch.float32
            )
            self.val_data_y = torch.tensor(
                val_data["target"].values, dtype=torch.float32
            )

    def train(self, epochs, consensus=None, lr_update=None):
        if lr_update != None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr_update)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            running_loss = 0.0
            optimizer.zero_grad()
            outputs = self(self.train_data_x)
            if self.ncl_lambda == None:
                loss = self.loss_function(outputs, self.train_data_y)
            else:
                loss = self.loss_function(
                    outputs, self.train_data_y, self.ncl_lambda, consensus
                )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            self.loss_curve.append(running_loss)

    def evaluate(self):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            outputs = self(self.val_data_x)
            loss = self.loss_function(outputs, self.val_data_y)
            val_loss += loss.item()

        return val_loss

    def predict(self, X):
        # if X already is not a tensor
        if not torch.is_tensor(X):
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        else:
            X_tensor = X

        self.eval()
        with torch.no_grad():
            output = self(X_tensor)

        return output.numpy()


class BaseLearnerNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        activation,
        loss_function,
        dropout_rates=None,
        lr=0.001,
        ncl_lambda=None,
        weight_decay=None,
    ):
        super(BaseLearnerNN, self).__init__()

        # Check if the number of layers, activations, and dropout rates match
        if dropout_rates != None:
            assert len(hidden_sizes) == len(
                dropout_rates
            ), "Mismatch in parameter lengths"

        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(self.get_activation(activation))
        if dropout_rates != None:
            layers.append(nn.Dropout(p=dropout_rates[0]))

        # Add hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(self.get_activation(activation))
            if dropout_rates != None:
                layers.append(nn.Dropout(p=dropout_rates[i]))

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Define the sequential model
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.model = nn.Sequential(*layers)
        self.loss_function = loss_function
        self.lr = lr
        self.ncl_lambda = ncl_lambda
        self.loss_curve = []
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        else:
            raise NotImplementedError(f"{activation} activation not implemented")

    def setup_data_loaders(self, df_train_set, validation_size=None):
        self.train_loader = None
        self.val_loader = None

        if validation_size != None:
            train_data, val_data = train_test_split(
                df_train_set, test_size=0.2, random_state=42
            )
        else:
            train_data = df_train_set

        self.train_data_x = torch.tensor(
            train_data.drop(columns=["target"]).values, dtype=torch.float32
        )
        self.train_data_y = torch.tensor(train_data["target"].values, dtype=torch.long)

        if validation_size != None:
            self.val_data_x = torch.tensor(
                val_data.drop(columns=["target"]).values, dtype=torch.float32
            )
            self.val_data_y = torch.tensor(val_data["target"].values, dtype=torch.long)

    def train(self, epochs, consensus=None, lr_update=None):
        if lr_update != None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr_update)
        elif self.weight_decay != None:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            running_loss = 0.0
            optimizer.zero_grad()
            outputs = self(self.train_data_x)
            if self.ncl_lambda == None:
                loss = self.loss_function(outputs, self.train_data_y)
            else:
                loss = self.loss_function(
                    outputs, self.train_data_y, self.ncl_lambda, consensus
                )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            self.loss_curve.append(running_loss)

    def evaluate(self):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            outputs = self(self.val_data_x)
            loss = self.loss_function(outputs, self.val_data_y)
            val_loss += loss.item()

        return val_loss

    def predict(self, X):
        # if X already is not a tensor
        if not torch.is_tensor(X):
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        else:
            X_tensor = X

        self.eval()
        with torch.no_grad():
            output = self(X_tensor)
            _, predicted_class = torch.max(output, 1)

        return predicted_class.numpy()


def cyclic_cosine_annealing_lr(lr, T_max, eta_min=0, last_epoch=-1):
    if last_epoch == 0:
        return lr

    if last_epoch % (2 * T_max) < T_max:
        return (
            eta_min
            + (lr - eta_min)
            * (1 + torch.cos(torch.tensor(math.pi * last_epoch / T_max)))
            / 2
        )
    else:
        return (
            eta_min
            + (lr - eta_min)
            * (1 + torch.cos(torch.tensor(math.pi * (last_epoch - T_max) / T_max)))
            / 2
        )
