import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import random
import os
import seaborn as sns

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


def load_data(where  = "./flowers" ):
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_data_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    
    return trainloader , vloader, testloader

def model_setup(structure = 'vgg16', dropout = 0.2, hidden_layer1, lr):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print('There is only vgg16. Please re run the module and enter only vgg16!')

    for param in model.parameters():
        param.requires_grad = False

    in_features_of_pretrained_model = model.classifier[0].in_features

    # alter the classifier so that it has 102 out features (i.e. len(cat_to_name.json))
    number_of_flower_classes = len(train_data.classes)
    classifier = nn.Sequential(nn.Linear(in_features=in_features_of_pretrained_model, out_features=hidden_layer1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=dropout),
                               nn.Linear(in_features=hidden_layer1, out_features=number_of_flower_classes, bias=True),
                               nn.LogSoftmax(dim=1)
                              )

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    return model, criterion, optimizer

def train_model(model, optimizer, criterion, epochs, trainloader, power):
        
    if power == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    print_every = 20  

    for e in range(epochs):
        step = 0
        running_train_loss = 0
        running_valid_loss = 0

        for images, labels in trainloader:
            step += 1

            model.train()

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           
            outputs = model(images)           
            train_loss = criterion(outputs, labels)            
            train_loss.backward()            
            optimizer.step()

            running_train_loss += train_loss.item()

            if step % print_every == 0 or step == 1 or step == len(trainloader):
                print("Epoch: {}/{} Batch % Complete: {:.2f}%".format(e+1, epochs, (step)*100/len(trainloader)))


        model.eval()
        with torch.no_grad():

            running_accuracy = 0
            running_valid_loss = 0
            for images, labels in vloader:                
                images, labels = images.to(device), labels.to(device)                
                outputs = model(images)

                valid_loss = criterion(outputs, labels)
                running_valid_loss += valid_loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


        average_train_loss = running_train_loss/len(trainloader)
        average_valid_loss = running_valid_loss/len(vloader)
        accuracy = running_accuracy/len(vloader)
        print("Train Loss: {:.3f}".format(average_train_loss))
        print("Valid Loss: {:.3f}".format(average_valid_loss))
        print("Accuracy: {:.3f}%".format(accuracy*100))

def save_checkpoint(path='checkpoint.pth', structure ='vgg16', hidden_layer1=120, dropout=0.2, lr=0.001, epochs=5):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing

    This function saves the model at a specified by the user path

    '''
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)


def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases

    '''
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = model_setup(structure , dropout, hidden_layer1, lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])        
        

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


trainloader, vloader, testloader = load_data(where)


model, optimizer, criterion = model_setup(structure, dropout, hidden_layer1, lr)


train_model(model, optimizer, criterion, epochs, trainloader, power)


save_checkpoint(path, structure, hidden_layer1, dropout, lr, epochs)


print("All Set and Done. The Model is trained")