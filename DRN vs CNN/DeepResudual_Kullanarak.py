import torch
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import time


#%% Read İmages

def read_images(path ,num_img):
    array = np.zeros([num_img ,64*32])
    i = 0
    
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path ,mode = "r")
        data = np.asarray(img ,dtype="uint8")
        data = data.flatten() 
        array[i ,:] = data
        
        i += 1
    return array

#%% train load
    
# negative 43390
train_negative_path = r"C:\Users\Mustafa\Desktop\MY_Working\Deep Learning\DRN VS CNN\LSIFIR\Classification\Train\neg"
train_negative_size = 43390
train_negative_array = read_images(train_negative_path , train_negative_size)

x_train_negative_tensor = torch.from_numpy(train_negative_array) # numpy arrayından torch arrayınıne çevirir
print("x_train negative_tensor size :" ,x_train_negative_tensor.size())

y_train_negative_tensor = torch.zeros(x_train_negative_tensor.size()[0] , dtype = torch.long)
print("y_trian negative_tensor size : " ,y_train_negative_tensor.size())

# positive 10208
train_positive_path = r"C:\Users\Mustafa\Desktop\MY_Working\Deep Learning\DRN VS CNN\LSIFIR\Classification\Train\pos"
train_positive_size = 10208
x_train_positive_array = read_images(train_positive_path ,train_positive_size)

x_train_positive_tensor = torch.from_numpy(x_train_positive_array)
print("x_train positive_tensor size :" ,x_train_positive_tensor.size())

y_trian_positive_tensor = torch.ones(x_train_positive_tensor.size()[0] ,dtype = torch.long)
print("y_train positivi_tensor size : ",y_trian_positive_tensor.size())

# concat
x_train = torch.cat((x_train_negative_tensor ,x_train_positive_tensor) ,0)
y_train = torch.cat((y_train_negative_tensor , y_trian_positive_tensor) ,0)

#%% Test load 

# Negative 22050
test_negative_path = r"C:\Users\Mustafa\Desktop\MY_Working\Deep Learning\DRN VS CNN\LSIFIR\Classification\Test\neg"
train_negative_size = 22050
train_negative_array = read_images(test_negative_path , train_negative_size)

x_test_negative_tensor = torch.from_numpy(train_negative_array)
print("x_test negative_tensor size :" ,x_test_negative_tensor.size())

y_test_negative_tensor = torch.zeros(x_test_negative_tensor.size()[0] , dtype = torch.long)
print("y_test negative_tensor size : " ,y_test_negative_tensor.size())

# Positive 5944
test_positive_path = r"C:\Users\Mustafa\Desktop\MY_Working\Deep Learning\DRN VS CNN\LSIFIR\Classification\Test\pos"
test_positive_size = 5944
x_test_positive_array = read_images(test_positive_path ,test_positive_size)

x_test_positive_tensor = torch.from_numpy(x_test_positive_array)
print("x_test positive_tensor size :" ,x_test_positive_tensor.size())

y_test_positive_tensor = torch.ones(x_test_positive_tensor.size()[0] ,dtype = torch.long)
print("y_test positivi_tensor size : ",y_test_positive_tensor.size())

# concat
x_test = torch.cat((x_test_negative_tensor ,x_test_positive_tensor) ,0)
y_test = torch.cat((y_test_negative_tensor , y_test_positive_tensor) ,0)

#%% virusalize

plt.imshow(x_train[50800 , :].reshape(64 ,32) ,cmap="gray")


# %% 
num_classes = 2
# Hyper parameters
num_epochs = 100
batch_size = 2000
learning_rate = 0.0001


import torch.utils.data 
train = torch.utils.data.TensorDataset(x_train ,y_train) # pytorch x_train ve y_tain setimizi kendi algoritmaları ile atıyor 
trainloader = torch.utils.data.DataLoader(train ,batch_size = batch_size ,shuffle = True) # shuffle = karıştır

test = torch.utils.data.TensorDataset(x_test ,y_test)
testloader = torch.utils.data.DataLoader(test ,batch_size = batch_size , shuffle = False)




#%% DRN Model Pytorch



def conv3x3(in_planes ,out_planes ,stride = 1): #stride filitreni basamak atlama sayısı
    return nn.Conv2d(in_planes ,out_planes ,kernel_size = 3 ,stride = stride ,padding = 1 ,bias = False)
    # kernel size = 3x3 olduğu için 3 , padding veri setimizin küçülmesini engelliyen köşelerini 0 atma işlemi

def conv1x1(in_planes ,out_planes ,stride = 1):
    return nn.Conv2d(in_planes ,out_planes ,kernel_size = 1 ,stride = stride ,bias = False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self ,inplanes ,planes ,stride = 1 ,downsample = None):
        # downsaple bizim DRN ile oluşan veri ile normal Cnn yapısıyla oluşan verilerin toplamında oluşacak olan boyut uyuşmazlıgını düzeltmek için kullanacağımız bir şecenektir
        super(BasicBlock ,self).__init__()
        self.conv1 = conv3x3(inplanes ,planes ,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(0.9)
        self.conv2 = conv3x3(planes ,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        
    def forward(self ,x):
        identity = x # identity bizim DRN ile oluuşturuduğumuz direk verinin relu üzerine iletildiği yapı 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        
        if self.downsample is not None:
            identity = self.downsample(x) # burada dönen olay biraz farklı gibi forumda araştıracağım 
            # mantıgını anladım ama arkada tam olarak ne gibi işlem yapıtyor boyut eşitlemek için ? 
            
        out = out + identity
        out = self.relu(out)
        return out
    
    
    
class ResNet(nn.Module):
    
    def __init__(self ,block ,layers ,num_classes = num_classes):
        super(ResNet ,self).__init__()
        
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1 ,64 ,kernel_size = 7 ,stride = 2 ,padding = 3  ,bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3 ,stride = 2 ,padding = 1)
        self.layer1 = self._make_layer(block ,64 ,layers[0] ,stride = 1) # layers boyutlarını dizi şeklinde veceğimiz için indis belirttik
        self.layer2 = self._make_layer(block ,128 ,layers[1] ,stride = 2)
        self.layer3 = self._make_layer(block ,256 ,layers[2] ,stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1 ,1)) # diğer poolardan farklı olarak biz çıktı boyutunmu veriyoruz ve o içierisindeki alğoritmalardan yararlanarak bize o boyutta çıktı dönderiyor
        self.fc = nn.Linear(256*block.expansion ,num_classes)
        
        for m in self.modules():
            if isinstance(m ,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight , mode = "fan_out" ,nonlinearity = "relu")
            elif isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight ,1)
                nn.init.constant_(m.bias ,0)
                
                
        
    #make_layer blocklarımızı birleştirme görevini yapar
    def _make_layer(self ,block ,planes ,blocks ,stride = 1): # blocks = block sayısı
       downsample = None
       if stride != 1 or self.inplanes != planes*block.expansion:
           downsample = nn.Sequential(# biz burda bir sequential oluşturarak girdi boyutlarımızı eşitliyoruz
               conv1x1(self.inplanes ,planes*block.expansion ,stride),
               nn.BatchNorm2d(planes*block.expansion))
           
       layers = []
       layers.append(block(self.inplanes ,planes ,stride ,downsample))
       self.inplanes = planes*block.expansion
          
       for _ in range(1 ,blocks):
           layers.append(block(self.inplanes ,planes))
     
       return nn.Sequential(*layers) # * işaretini listeyi içine düzgün şekilde göndermesi için yapıyoruz
   
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x
    
    
model = ResNet(BasicBlock, [2,2,2])
    

#%% 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters()  ,lr = learning_rate)

#%% Train


loss_listy = []
train_acc = []
test_acc = []

total_step = len(trainloader)

for epoch in range(num_epochs):
    for i, (images ,labels) in enumerate(trainloader):
        
        images = images.view(batch_size,1,64,32)
        images = images.float()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        #backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if i % 2 == 0:
            print("epoch: {} {}/{}".format(epoch ,i ,total_step))
            
    #train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images ,labels = data 
            images = images.wiew(batch_size,1 ,64 ,32)
            images = images.float()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data ,1)
            total += label.size(0)
            correct += (predicted == labels).sum().item()
            
    print("Accuracy Train %d %%"%(100*correct/total))
    train_acc.append(100*correct/total)
    
    
    #test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images ,labels = data 
            images = images.wiew(batch_size,1 ,64 ,32)
            images = images.float()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data ,1)
            total += label.size(0)
            correct += (predicted == labels).sum().item()
            
    print("Accuracy Test %d %%"%(100*correct/total))
    test_acc.append(100*correct/total)
    
    loss_list.append(loss.item())
    
#%% visualize

fig, ax1 = plt.subplots()
plt.plot(loss_list,label = "Loss",color = "black")
ax2 = ax1.twinx()
ax2.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()


    
    
    
    
    
    

        
        
        
        