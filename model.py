import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

#Generate Synthetic Graph Data for PDE (Hamilton-Jacobi-Bellman Equation)
def generate_graph(num_nodes=1000):
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5))  # Random graph structure
    x = torch.rand((num_nodes, 1))
    y = x**2  # PDE solution approximation (HJB example: quadratic form)
    return Data(x=x, edge_index=edge_index, y=y)

class GNN_PDE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_PDE, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

'''
Basic Model Training
'''
num_nodes = 1000
graph_data = generate_graph(num_nodes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNN_PDE(in_channels=1, hidden_channels=32, out_channels=1).to(device)
graph_data = graph_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    output = model(graph_data)
    loss = criterion(output, graph_data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

''' Evaluate and Visualize Results'''
predicted_solution = model(graph_data).cpu().detach().numpy()
true_solution = graph_data.y.cpu().detach().numpy()
error = np.abs(predicted_solution - true_solution).mean()
print(f'Final Mean Absolute Error: {error}')
