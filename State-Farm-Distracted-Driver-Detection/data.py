import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
import helper
import pandas as pd




def output_file(path, predict):
    data_dir = path +'/test/1'
    target = os.listdir(data_dir)
    submission = pd.DataFrame(predict, columns = ["c0","c1","c2",	"c3",	"c4",	"c5",	"c6",	"c7",	"c8",	"c9"] , index = target)
    submission.to_csv("submission.csv")
    






def get_data(path, device = None):

    print("image loading ...........")
    '''
    path
    |---train
    |     |---c0
    |     |    | --- image1
    |     |
    |---test
    |     |---image 1
    '''

    data_dir = path
    train_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_dir + '/train/train',  
                                        transform=train_transforms)  

    val_data = datasets.ImageFolder(data_dir + '/train/val',  
                                        transform=val_transforms)                                     
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    
    # #Data Loading
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = False)
   
    


    return trainloader, valloader, testloader

if __name__ == "__main__":
    get_data("./../../data",)