import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

np.random.seed(42)

# Generate some data
X = np.random.rand(100, 1) # create an array of the given shape, uniform distributed over [0,1)
y = 2 * X + 3 + 0.1 * np.random.randn(100, 1)

# Convert to PyTorch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    model.train()

    y_pred = model(X)

    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 ==0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred = model(X)


plt.scatter(X.numpy(), y.numpy(), label='True data')
plt.plot(X.numpy(), y_pred.numpy(), color='r', label='Fitted data')
plt.legend()
plt.show()

    