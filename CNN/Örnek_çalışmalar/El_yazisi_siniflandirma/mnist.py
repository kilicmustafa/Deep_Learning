#import list
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% load dataset

train = pd.read_csv("mnist_train.csv")
print(train.shape)
print(train.head(5))

test = pd.read_csv("mnist_test.csv")
print(test.shape)
print(train.head(5))


#%% X- Y separation
y_train = train["label"]
x_train = train.drop(labels = ["label"] ,axis = 1)

y_test = test["label"]
x_test = test.drop(labels = ["label"] ,axis = 1)
print(y_train.head(3))
print(x_train.head(3))
print(y_test.head(3))
print(x_test.head(3))

plt.figure()
sns.countplot(y_train , palette = "icefire")
plt.title("train y_head class variable counts")
print(y_train.value_counts())
plt.show()

plt.figure()
sns.countplot(y_test)
plt.title("test y_head class variable counts")
print(y_test.value_counts())
plt.show()

plt.figure()
img = np.array(x_train.iloc[9])
img = img.reshape((28 ,28))
plt.imshow(img , cmap= "gray")
plt.axis("off")
plt.show()


#%% Normalization , Reshape and Label Encoding

#Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0
print("x_train shape : ",x_train.shape)
print("x_test shape : " , x_test.shape)

#Reshape
x_train = x_train.values.reshape( -1 ,28,28 ,1 )
x_test = x_test.values.reshape( -1 ,28,28 ,1 )
print("x_train shape : " , x_train.shape)
print("x_test shape : ",x_test.shape)





#%% Train - Validation split
from sklearn.model_selection import train_test_split
x_train ,x_val ,y_train , y_val = train_test_split(x_train ,y_train , random_state = 3 ,test_size = 0.1 )
print("x_train shape : " , x_train.shape)
print("y_train shape : " ,y_train.shape)
print("x_val shape : " ,x_val.shape)
print("y_val shape : " ,y_val.shape)



#%% Label Encoding

#Label encoding Keras
from keras.utils.np_utils import to_categorical
y_train = to_categorical( y_train ,num_classes = 10)
y_val = to_categorical(y_val ,num_classes = 10)
y_test = to_categorical(y_val , num_classes = 10)

#%% Create Model



from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D ,Activation ,Dropout ,Flatten ,Dense
from keras.preprocessing.image import ImageDataGenerator 
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(filters = 16 , kernel_size = (3 ,3) ,input_shape = (28 , 28 ,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32 , kernel_size = (3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64 , kernel_size = (3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10 )) # değisken sayısı
model.add(Activation("softmax")) #kategori sayısı fazla olduğu için


optimizer = Adam(lr = 0.001 ,beta_1 = 0.9 ,beta_2 =0.999 )
model.compile(optimizer = optimizer,
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

batch_size = 32
epochs = 10 # 
#%% Data generation Train-Test


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # resimleri çevirir 6 yi 9  yapabilir
        vertical_flip=False)  # basamağı değiştirdiği için kullanılmaz )

datagen.fit(x_train)
hist= model.fit_generator(datagen.flow(x_train ,y_train , batch_size = batch_size),
                              validation_data = (x_val ,y_val),
                              epochs =epochs,
                              steps_per_epoch = 1600 // batch_size)



#%% Model Save
model.save_weights("save_model_1.h5") 

#%% Save History
import pandas as pd
import json 

hist_df = pd.DataFrame(hist.history)
with open("hist_save.json" ,"w") as f:
    hist_df.to_json(f)
    
#%% Load History
    
with open("hist_save.json") as json_file:
    h = json.load(json_file)
    
df = pd.DataFrame(h)

print(df)

plt.plot(df["loss"], label = "Train Loss")

plt.plot(df["val_loss"], label = "Validation Loss")

plt.legend()

plt.show()

plt.plot(df["accuracy"], label = "Train Loss")

plt.plot(df["val_accuracy"], label = "Validation Loss")

plt.legend()

plt.show()

