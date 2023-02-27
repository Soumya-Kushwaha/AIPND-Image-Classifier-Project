import torch 
import numpy as np
from torch import optim,nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import argparse
import json
from os.path import isdir


def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type = str, default = 'flowers') 

        parser.add_argument('--learning_rate', type = float, default =0.001) 

        parser.add_argument('--epochs', type = int, default = 5) 
        parser.add_argument('--arch', type = str, default = 'vgg16') 
        parser.add_argument('--save_dir', type = str, default = 'save_directory') 
        parser.add_argument('--gpu', action="store_true") 
        parser.add_argument('--hidden_units', type=int)
            
        args= parser.parse_args()
        return args
in_arg = arg_parser()


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)



train_loader = torch.utils.data.DataLoader(train_datasets,batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets,batch_size=64,shuffle=False)
test_loader = torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle=False)

hidden_layer_units = in_arg.hidden_units
if type(hidden_layer_units) == type(None): 
        hidden_layer_units = 4086

def check_gpu():
    if(in_arg.gpu):
        if(torch.cuda.is_available):
            device = torch.device("cuda")
        else:
            print("Cuda is not available on system using cpu")
    else:
        device = torch.device("cpu")
    return device    


input_units=0

if (in_arg.arch=='vgg16'):
    model= models.vgg16(pretrained=True)
    model.name = "vgg16"
    input_units = 25088    
elif (in_arg.arch=='densenet161'):
    exec("model= models.{}(pretrained=True)".format(in_arg.arch))
    input_units = 2208    
    
elif (in_arg.arch=='alexnet'):
    exec("model= models.{}(pretrained=True)".format(in_arg.arch))
    input_units = 9216

else:
    print("Please provide a valid classifier model for training!")
    
      

     
    
for p in model.parameters():
    p.requires_grad = False     
  
 
    
print("\n CNN Model Architecture for classifier for training : {}".format(in_arg.arch))
          
 
def set_classifier():
    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_units, hidden_layer_units, bias=True)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.4)),
                            ('fc2', nn.Linear(hidden_layer_units, 102, bias=True)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier =classifier


def train_model():
        
    epochs = in_arg.epochs
    steps = 0
    print("\nModel training has been started..........It may take a while:")

    for e in range(epochs):
        cmmulative_loss = 0
        model.train() 
        print("\n<--------------------------------------- Round {} Training Started --------------------------------------->\n".format(e+1))
        
        
        for inputs, labels in train_loader:
            
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cmmulative_loss += loss.item()

            if steps % 30 == 0:
               model.eval()

               with torch.no_grad():
                    data = validation_test(model, valid_loader, criterion)
                    valid_loss, accuracy = data
               print("Training Loss: {:.4f}  ".format(cmmulative_loss/steps), end=" ")
               print("Validation Loss: {:.4f}  ".format(valid_loss/len(test_loader)), end=" ")
               print("Validation Accuracy: {:.4f}".format(accuracy/len(test_loader)))

               running_loss = 0
               model.train()
    print("\n<--------------------------------------- Model Trained Successfully --------------------------------------->\n")


def validation_test(model, test_loader, criterion):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(test_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        loss = criterion(output, labels)
        test_loss += loss.item()
        
        ps = torch.exp(output)
        equal = labels.data == ps.max(dim=1)[1]
        accuracy += torch.mean(equal.type(torch.FloatTensor))
    
    return test_loss, accuracy
    

def final_validation():
    equal = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pd = torch.max(outputs.data, 1)
            total += labels.size(0)
            equal += (pd == labels).sum().item()

    f_accuracy=100 * equal / total
    print("Final Testing Accuracy Achieved :",f_accuracy )
        
        
        
def save_checkpoint():
    
    file_path = in_arg.save_dir + 'checkpoint.pth'
            
    if type(in_arg.save_dir)== type(None):
        print("Please provide directory for saving checkpoint with trained model")

    else:
        if isdir(in_arg.save_dir):

            model.class_to_idx = train_datasets.class_to_idx
            checkpointData = { 'arch':model.name,
                            'classifier': model.classifier,
                            'epochs':in_arg.epochs,
                            'dropout':0.5,
                            'class_to_idx':model.class_to_idx,
                            'state_dict': model.state_dict()}
            torch.save(checkpointData, file_path )
            print("\n<--------------------------------------- Checkpoint Data Saved Successfully --------------------------------------->\n")

        else:
            print("Wrong directory----------- Unable to save the model")


            
            
            
            

set_classifier()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), in_arg.learning_rate)
device=check_gpu()
model.to(device)

train_model()
final_validation()
save_checkpoint()
