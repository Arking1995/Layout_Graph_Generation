import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(33, 128)
        self.conv2 = GCNConv(128, 256)
        self.linear1 = torch.nn.Linear(256, 10)

    def forward(self, data):
        batch_size = list(data.ptr.shape)[0] - 1

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = F.relu(x)

        if batch_size > 1:
            for i in range(batch_size):
                g_x = x[data.ptr[i].cpu().numpy(): data.ptr[i+1].cpu().numpy(), :]
                if i == 0:
                    h_G = g_x.sum(dim=0).unsqueeze(0)
                else:
                    tmp = g_x.sum(dim=0).unsqueeze(0)
                    h_G = torch.cat((h_G, tmp), 0)
        else:    
            h_G = x.sum(dim=0).unsqueeze(0)         

        y_p1 = self.linear1(h_G)
        return F.log_softmax(y_p1)