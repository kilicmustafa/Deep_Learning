#%% import libary
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Conv2D ,MaxPooling2D , Activation ,Dense ,Dropout ,Flatten ,BatchNormalization
from keras.utils import to_categorical

#%% Load And Preprocessing
def load_and_preprocessing(data_path):
    data = pd.read_csv(data_path)
    data = np.array(data)
    np.random.shuffle(data) #Veri setimizi rasgele karıştırır
    x = data[:,1:].reshape(-1 ,28,28 ,1) /255.0
    y = data[: ,0].astype(np.int32)
    y = to_categorical(y , num_classes = len(set(y)) ,dtype = "int32")

    return x ,y

train_data_path = "mnist_train.csv"
test_data_path = "mnist_test.csv"

x_train ,y_train = load_and_preprocessing(train_data_path)
x_test , y_test  = load_and_preprocessing(test_data_path)

#%% Visualize 
index = 89
vis = x_train.reshape(60000 ,28,28)
plt.imshow(vis[index ,: ,:])
plt.axis("off")
plt.legend()
plt.show()

print(np.argmax(y_train[index]))#one-hot dönuşumu yaptık ve o index'deki değerin karşılığını almak için yaparız

#%% CNN model 

numberOfClass = y_train.shape[1] # one-hot dönüşümünden sonra kategori sayımızı bu şekilde alabiliriz

model = Sequential()

model.add(Conv2D(filters = 16 ,kernel_size = (3,3) ,input_shape = (28 ,28, 1) ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64 ,kernel_size = (3 ,3) ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units = 256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(units = numberOfClass)) 
model.add(Activation("softmax"))

model.compile(optimizer = "adam",
              metrics = ["accuracy"],
              loss = "categorical_crossentropy")


epochs = 3
batch_size = 4000
hist = model.fit(x_train ,y_train , validation_data = (x_test ,y_test) ,epochs = epochs ,batch_size = batch_size)


#%% Model Save
model.save_weights("model_saved_2.h5")


#%% evulation

print(hist.history.keys())

plt.plot(hist.history["loss"] ,label = "Train Loss")
plt.plot(hist.history["val_loss"] ,label = "Val_loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"] ,label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"] , label = "Val accuracy")
plt.legend()
plt.show()


#%% Save history

hist_df = pd.DataFrame(hist.history)

with open("hist_save_2.json" ,"w") as f :
    hist_df.to_json(f)
    
    
#%% Load History 
    
import json 
with open("hist_save_2.json") as json_file:
    h = json.load(json_file)
    
    
df = pd.DataFrame(h)
plt.plot(df["loss"] , label = "Train Loss")
plt.plot(df["val_loss"] ,label = "Val loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(df["accuracy"] ,label = "Train accuracy")
plt.plot(df["val_accuracy"] ,label = "Val Accuracy") 
plt.legend()
plt.show()

   
    
