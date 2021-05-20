import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1  = nn.Conv2d(3,32,kernel_size = (3, 3))
        self.mp1    = nn.MaxPool2d(3)
        self.conv2  = nn.Conv2d(32,64,kernel_size = (3, 3))
        self.mp2    = nn.MaxPool2d(3)
        self.fc1    = nn.Linear(24*24*64, 128)
        self.fc2    = nn.Linear(128,32)
        self.fc3    = nn.Linear(32,10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)
        return x





class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResidualBlock(in_channel = 3  , out_channel = 16 , stride = 1)
        self.layer2 = ResidualBlock(in_channel = 16 , out_channel = 32 , stride = 2)
        self.layer3 = ResidualBlock(in_channel = 32 , out_channel = 32 , stride = 2)
        self.layer4 = ResidualBlock(in_channel = 32 , out_channel = 64 , stride = 2)
        self.layer5 = ResidualBlock(in_channel = 64 , out_channel = 32 , stride = 2)
        self.fc1    = nn.Linear(32*14*14 ,48)
        self.fc2    = nn.Linear(48 ,10)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x , dim = 1)



class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = (3,3), padding = 1, stride = stride)
        self.bn1   = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = (3,3), padding = 1, stride = 1)
        self.bn2   = nn.BatchNorm2d(out_channel)
        self.down  = nn.Conv2d(in_channel, out_channel, kernel_size = (1,1), stride = stride)
        
    def forward(self, inputs):
        x  = self.conv1(inputs)
        x  = self.bn1(x)
        x  = self.conv2(x)
        x  = self.bn2(x)
        x_ = self.down(inputs)
        return F.relu(x + x_) 
        



class Framework():
    def __init__(self, args):
        self.lr             = args.lr
        self.device         = torch.device(args.device)
        self.epoch          = args.epoch
        self.bs             = args.batch_size
        self.model          = ResNet().to(self.device)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.loss_fn        = nn.CrossEntropyLoss()
        self.path           = args.train_data_path
        self.train_loader, self.val_loader, self.test_loader = get_data(self.path, self.device)
       
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def show_model(self):
        print(self.model)
        print("Trainable Parameters: %d"%sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def validation(self):
        self.model.eval()
        for val_x, val_label in self.train_loader:
            val_x, val_label = val_x.to(self.device) , val_label.to(self.device)
            val_y = self.model(val_x)
            val_loss = self.loss_fn(val_y, val_label)

            val_pred = torch.max(val_y.cpu(), 1 )[1].data.numpy()
            val_acc = float((val_pred == val_label.cpu().detach().numpy()).astype(int).sum()) / float(val_label.cpu().detach().numpy().size)

            return val_loss, val_acc


    def test(self):
        self.model.eval()
        result = []
        for test_x, _ in self.test_loader:
            test_x = test_x.to(self.device)
            test_y = self.model(test_x)

            for y in test_y.cpu().detach().numpy():
                result.append(y)
        print(np.array(result).shape)
        return np.array(result)


    def train(self):
        self.model.train()
        print("-"*50)
        print('start training....')
        print("-"*50)
        for epoch in range(self.epoch):
            loss_mean = 0.
            for steps,(x, label) in enumerate(self.train_loader):
                
                x, label = x.to(self.device) , label.to(self.device)

                y = self.model(x)
                loss = self.loss_fn(y , label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_mean += loss.cpu().item()

                if((steps+1)%50 == 0):
                    train_pred = torch.max(y.cpu(), 1 )[1].data.numpy()
                    train_acc = float((train_pred == label.cpu().detach().numpy()).astype(int).sum()) / float(label.cpu().detach().numpy().size)
                    loss_mean /= 50
                    val_loss, val_acc = self.validation()
                    self.model.train()

                    print("Epoch:  %3d  | Steps: %3d  | loss: %.4f | acc: %f | val loss: %.4f | val_acc: %.4f |" %(epoch+1, steps+1, loss_mean , train_acc, val_loss, val_acc))
                    # print("Epoch:  %3d  | Steps: %3d  | loss: %.4f | acc: %f " %(epoch+1, steps+1, loss_mean , train_acc))

                    loss_mean = 0.
                    self.model.train()
        