import torch
import torch.nn.functional as F
import pandas as pd
class net_phy(torch.nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(net_phy, self).__init__()
        self.hidden1 = torch.nn.Linear(input, hidden1)
        self.hidden2 = torch.nn.Linear(hidden1, hidden2)
        self.predict = torch.nn.Linear(hidden2, output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        physical_info=x
        if self.training:
            return x
        else:
            return x,physical_info

class net_concat(torch.nn.Module):
    def __init__(self,input,hidden1,hidden2,output):
        super(net_concat,self).__init__()
        self.hidden1=torch.nn.Linear(input,hidden1)
        self.hidden2=torch.nn.Linear(hidden1,4)
        self.predict=torch.nn.Linear(hidden2,output)


    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=torch.cat([x,self.physical_info],1)
        x=self.predict(x)
        return x

    def add_physical_info(self,physical_info):
        self.physical_info=physical_info