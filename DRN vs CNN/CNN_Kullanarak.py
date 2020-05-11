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
        data = np.asarray(img ,dtype = "uint8")
        data = data.flatten()
        array[i, :] = data 
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


#%% CNN Model PyTorch

#Hyperparameter

num_epochs = 5000
num_classes = 2 
batch_size = 8933
learning_rate = 0.00001

class Net(nn.Module):
    
    def __init__(self):
        super(Net ,self).__init__()
        
        self.conv1 = nn.Conv2d(1 ,10 ,5)
        self.pool = nn.MaxPool2d(2 ,2)
        self.conv2 = nn.Conv2d(10 ,16 ,5)
        
        self.fc1 = nn.Linear(16*13*5 ,520)
        self.fc2 = nn.Linear(520 ,130)
        self.fc3 = nn.Linear(130,num_classes)
    def forward(self ,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.wiew(-1 ,16 *13*5) # flatten yöntemi 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        
        return x
    

import torch.utils.data 
train = torch.utils.data.TensorDataset(x_train ,y_train) # pytorch x_train ve y_tain setimizi kendi algoritmaları ile atıyor 
trainloader = torch.utils.data.DataLoader(train ,batch_size = batch_size ,shuffle = True) # shuffle = karıştır

test = torch.utils.data.TensorDataset(x_test ,y_test)
testloader = torch.utils.data.DataLoader(test ,batch_size = batch_size , shuffle = False)

net = Net()
# gpu için 
# net = net().to(device)

#%% loss and optimizer

# loss
criterion = nn.CrossEntropyLoss() 

#optimizer
import torch.optim as optim 

optimizer = optim.SGD(net.paramesters() ,lr = learning_rate , momentum = 0.8) #momentum öğrenme hızını etkileyen unsur


#%% train network 

start = time.time()
train_acc = []
test_acc = []
loss_list = []
use_gpu = False # sonradan gpu kullanıyor isek bunu true yapacağız

for epoch in range(num_epochs): # epoch sayım kadar dönmesi için for döngüsü açıyoruz
    for i ,data in enumerate(trainloader ,0):
        # enumerate(sıralanacak_deger, baslangıc_sayısı) liste dönderir
        inputs ,labels = data 
        inputs = inputs.wiew(batch_size ,1 ,64 ,32)# .wiew rashape ile aynı görevi görür
        inputs = inputs.float() # input değişkenimizi floata çeviriyoruz
        
        # use gpu
        """if use_gpu:
            if torch.cuda.is_available():
                inputs ,labels = inputs.to(device) ,labels.to(device) # değerlerimizi gpu ya gönderiyoruz"""
                
                
        # zero gradient 
        optimizer.zero_grad() # pytorch kullanırken gradları sıfırlamamız gerekiyor aksi halde her dönüşte gradlar toplanım modelimizi bozacaktır
        
        # forward 
        outputs = net(inputs)
    
        # loss
        loss = criterion(outputs ,labels)
        
        # back
        loss.backward()
        
        # update weights 
        optimizer.step()
        
        
        
    # Test için her epoch'da yapılacak işlemler 
    correct = 0 
    total = 0 
    with torch.no_grap():# epoch'u durdurup buraya gir 
        for data in testloader:
            images, labels = data
            images = images.wiew(batch_size ,1 ,64 ,32)
            images = images.float()
            
            # gpu use
            """if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)"""
            
            outputs = net(images) # pred
            
            _ , predicted = torch.max(outputs.data ,1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # item fonksiyonu bize dönen true false vektorunu sayısala cavirmemizi sağlar 
            # doru hesapladıpımız değerlerin sayısını almak için yopıyoruz 
            
            
    acc1_test = 100*correct/total
    print("accuracy test : " ,acc1_test)
    test_acc.append(acc1_test)
    
    
    
    
    # Train için her opoch'da yapılacak işlemler
     
    correct = 0
    total = 0 
    with torch.no_grap():
        for data in trainloader:
            images ,labes = data
            images = images.wiew(batch_size , 1 ,64 ,32)
            images = images.float()
            
            outputs = net(images)
            
            _ ,predicted = torch.max(outputs.data ,1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().iter()
            
            
    acc2_train = 100*correct/total
    print("accuracy train :" ,acc2_train)
    train_acc.append(acc2_train)
            
    
    
print("train and test are done.")
            

end = time.time()
process_time = (end - start)/60
print("proccess time :" ,process_time)

#%% visualize 

fig ,ax1 = plt.subplot()
plt.plot(loss_list ,label ="Loss" ,color ="black")

ax2 = ax1.twinx() # görselin sağında yeni bir eksen  doğrusu oluşturur şu değerlerin yazıların yazıldığı :)

ax2.plot(np.array(test_acc)/100 ,label = "Test acc" ,color ="green")
ax2.plot(np.array(train_acc)/100 ,label ="Train acc" ,color ="red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
# tight_layout, alt grafik parametrelerini şekil alanına sığacak şekilde otomatik olarak alt grafik parametrelerini ayarlar.,


plt.title("Loss vs Test Accuracy")
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
