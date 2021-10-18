import torch, os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from UrbanDataset import UrbanGraphDataset, graph_transform
from TrivialDataset import UrbanGraphDataset, graph_transform
from graphneural import GCN
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch_geometric.data import Batch


data_name = 'chicago'
loss_name = 'mse'

root = os.getcwd()
dataset = UrbanGraphDataset(root=os.path.join(root,'dataset', data_name), transform = graph_transform)
# loader = DataLoader(dataset, batch_size=1, shuffle=True)


# test_dataset = dataset[:30]
val_dataset = dataset[1600:]
train_dataset = dataset[:1600]
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)



device = torch.device('cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    #    factor=0.7, patience=5,
                                                    #    min_lr=0.00001)
entropyloss = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()

def train(epoch):
    model.train()
    loss_all = 0
    acc1 = 0
    acc2 = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out1 = model(data)
        posx = data.y1


        # Here we only test the posx of y, which is the second value of data.y of input to graph_transform(data):
        # implementation is line125 in TrivialDataset.py, change that assignment for future targets.
        pred1 = out1.argmax(dim=1)
        
        loss1 = entropyloss(out1, posx)

        loss = loss1
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()


   
        correct1 = (pred1 == posx).sum()
        acc1 += int(correct1)

    return acc1 / len(train_loader.dataset),  loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()
    acc1 = 0
    acc2 = 0

    for data in loader:
        data = data.to(device)
        out1 = model(data)
        pred1 = out1.argmax(dim=1)

        posx = data.y1

        correct1 = (pred1 == posx).sum()
        acc1 += int(correct1)

    return acc1 / len(loader.dataset)


best_val_acc = None
best_train_acc = None

for epoch in range(1, 1000):
    # lr = scheduler.optimizer.param_groups[0]['lr']
    lr = 0.01
    train_acc, loss = train(epoch)
    val_acc = test(val_loader)
    # scheduler.step(val_error)

    if best_val_acc is None or val_acc >= best_val_acc:
        best_val_acc = val_acc
        filn = os.path.join(root,'epoch',data_name + '_' + loss_name + "_val_best.pth")
        torch.save(model.state_dict(), filn)

    if best_train_acc is None or train_acc >= best_train_acc:
        best_train_acc = train_acc
        filn = os.path.join(root,'epoch',data_name + '_' + loss_name + "_train_best.pth")
        torch.save(model.state_dict(), filn)

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Train Accuracy: {:.7f},  Validation Accuracy: {:.7f}, '.format(epoch, lr, loss, train_acc, val_acc))
print('Best Val accuracy: {:7f}, train accuracy: {:7f}'.format(best_val_acc, best_train_acc))
