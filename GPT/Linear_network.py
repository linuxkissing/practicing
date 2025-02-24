import torch
import torch.nn as nn
import torch.optim as optim

class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.dropout_layer(y)
        return y
    
model = Linear(20, 20, 20)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(20, 20, requires_grad=True)
y = torch.randn(20, 20, requires_grad=True)

epoch = 0
max_epoch = 10000
while True:
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    if loss.item() < 1e-3:
        print('Final Epich{', epoch+1, '}')
        break

    if epoch+1 >= max_epoch:
        print('Final Epoch{', epoch+1, '}')
        break