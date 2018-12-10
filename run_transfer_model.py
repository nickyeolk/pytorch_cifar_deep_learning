import cv2
import src.week4_func as wk4
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
import src.build_model as mdl
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import torchvision.transforms.functional as TF

df_1 = wk4.just_dataframes('./data/cifar-10-batches-py/data_batch_1')
df = pd.concat([df_1,wk4.just_dataframes('./data/cifar-10-batches-py/data_batch_2')],axis=0)
df = pd.concat([df,wk4.just_dataframes('./data/cifar-10-batches-py/data_batch_3')],axis=0)
df = pd.concat([df,wk4.just_dataframes('./data/cifar-10-batches-py/data_batch_4')],axis=0)
df = pd.concat([df,wk4.just_dataframes('./data/cifar-10-batches-py/data_batch_5')],axis=0)
df_test = wk4.just_dataframes('./data/cifar-10-batches-py/test_batch')
X_all = df.drop('target',axis=1).values/255

X_train = X_all.reshape(-1,3,32,32,order='C')
y_train = df['target'].values
X_test = (df_test.drop('target',axis=1).values/255).reshape(-1,3,32,32,order='C')
y_test = df_test['target'].values

def transform_norm(dataarray):
    mean=np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1944, 0.2010])
    for i in range(3):
        dataarray[:,i,:,:] = (dataarray[:,i,:,:]-mean[i])/std[i]
    return newarray
#X_train = transform_norm(X_train)
#X_test = transform_norm(X_test)


class TensorDataset(Dataset):
    def __init__(self,tensor,labels):
        self.tensor=tensor
        self.labels=labels
        self.resize = transforms.Resize(size=(224,224))
    def __getitem__(self,index):
        out = self.tensor[index]
        out = TF.to_pil_image(out)
        out = self.resize(out)
        out = TF.to_tensor(out)
        #print('throwout',out.size(),self.labels.size())
        return out, self.labels[index]
    def __len__(self):
        return self.tensor.size(0)

train_data = TensorDataset(torch.from_numpy(X_train).float(),torch.from_numpy(y_train))
trainloader = DataLoader(train_data,batch_size=50,shuffle=True)
test_data = TensorDataset(torch.from_numpy(X_test).float(),torch.from_numpy(y_test))
testloader = DataLoader(test_data,batch_size=50,shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('initiate model, device is, ', device)
#net = mdl.Net()
net = models.resnet18(pretrained=True)
for param in net.parameters():
        param.requires_grad = False
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 10)
print('convert')
net = net.to(device)
#net.to(device)
print('model initiated')
# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train_model(net, trainloader, criterion, optimizer,f):
    for epoch in range(100):
        running_loss=0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500==499:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500),file=f)
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i+1, running_loss/500))
                running_loss = 0.0
    print('Finished Training')
    return net

def scoring(net, testloader, f):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total),file=f)

with open('transfer_learning_output.txt', 'a') as f:
    net = train_model(net,trainloader, criterion, optimizer, f)
    print('training complete', f)
    scoring(net, testloader, f)
    
torch.save(net.state_dict(), './model_saved.pt')
