import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import visualization as vis

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except Exception as e:
from tensorboardX import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x_0):
        x_0 = self.pool(F.relu(self.conv1(x_0)))
        x_1 = self.pool(F.relu(self.conv2(x_0)))
        x_2 = x_1.view(-1, 16 * 4 * 4)
        x_3 = F.relu(self.fc1(x_2))
        x_4 = F.relu(self.fc2(x_3))
        x_5 = self.fc3(x_4)
        return x_5

if __name__ == "__main__":
    log_dir = "./board"
    tb_writer = SummaryWriter(log_dir)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # simulate input and label
    input = torch.randn(1, 1, 28, 28)
    label = torch.ones(1).long()

    # log model graph
    vis.log_module(module  = net,
                  input    = input,
                  tb_writer= tb_writer)

    
    out_graph_1 = vis.log_grad_graph(module=net,
                       input =input,
                       log_name='graph_test_1',
                       log_dir=log_dir,
                       view=True)

    # log grad graph(not always work)
    # out_graph_2 = vis.log_grad_graph_from_trace(module =net,
    #                                             input  =input,
    #                                             log_name='graph_test_2',
    #                                             log_dir=log_dir,
    #                                             view   =True)


    # log step grad and weight
    for i in range(10):
        output = net(input)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        vis.log_grads(module    = net,
                      tb_writer = tb_writer, 
                      tb_index  = i,
                      tb_name   = "Net") 
