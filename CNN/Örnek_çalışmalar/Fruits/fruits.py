#libarys
from keras.models import Sequential 
from keras.layers import Conv2D , MaxPooling2D , Activation , Dropout ,Flatten ,Dense
from keras.preprocessing.image import ImageDataGenerator , img_to_array , load_img
import matplotlib.pyplot as plt
from glob import glob

#Dataset path

test_path = "fruits-360/Test/"
train_path = "fruits-360/Training/"


#İmgShow

img = load_img(train_path + "Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()


x = img_to_array(img)
print(x.shape)

className = glob(train_path + "/*")
numberOfClass = len(className)
print("NumberOfClass : " ,str(numberOfClass))


#%% CNN Model

model = Sequential()

model.add(Conv2D(32 , (3,3) ,input_shape = x.shape)) # filtreSayısı , FilitreBoyutu , dataBoyutu
model.add(Activation("relu"))
model.add(MaxPooling2D())#default değerini bırakıyoruz

model.add(Conv2D(32 ,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64 , (3 ,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten()) #Dataset uzatma
model.add(Dense(1024))#node sayısı
model.add(Activation("relu"))
model.add(Dropout(0.5))# node kapatım test etme oranı
model.add(Dense(numberOfClass)) # class sayısı / output
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])
batch_size = 32


#%% Data Generation - Train -Test

train_datagen = ImageDataGenerator(
    rescale = 1./255 # resim olduğu için 255 ile normalize ettik
    ,shear_range = 0.3 # random bir şekilde cevrip oluşturma
    ,horizontal_flip = True # yatay cevirip resim oluşturma
    ,zoom_range= 0.3 # zoom yapıp oluşturma
    )


test_datagen = ImageDataGenerator(rescale = 1./255) # test setimiz için sadece ölçeklendirme işlemi yapacağız


train_generator = train_datagen.flow_from_directory(#dosya içinde resim klasları onun içindede resimler var ise otomatik çalışır
    train_path # dosya yolumuz
    ,target_size = x.shape[:2] # resimlerimizin boyutu sonudaki 3 ' almadık sebebi altta başka bir parametrede belirticek olamamız
    ,batch_size = batch_size 
    ,color_mode = "rgb"
    ,class_mode = "categorical"
    )

test_generator = test_datagen.flow_from_directory(
    test_path
    ,target_size = x.shape[:2]
    ,batch_size = batch_size
    ,color_mode = "rgb"
    ,class_mode = "categorical"
    )


hist = model.fit_generator(
    generator = train_generator # fit edeceğimiz generator
    ,steps_per_epoch = 1600 // batch_size # her class içinde kaç resim epochs edeceğimiz gibi birşey
    ,epochs = 3 # epochs edilecek resim sayısı sanırsam bunu araştıracağım
    ,validation_data = test_generator # fit edeceğimiz test seti
    ,validation_steps = 800 // batch_size)

#%% Model Save
model.save_weights("model_save_1.h5")

#%% model evaluation
print(hist.history.keys()) #hist değişkininde kullanabileceğimiz değerleri alıyoruz
plt.plot(hist.history["loss"] ,label = "Train Loss")
plt.plot(hist.history["val_loss"] , label = "Validation Loss")
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"] , label = "Train acc")
plt.plot(hist.history["val_accuracy"] ,label = "Test acc")
plt.show()


#%% save history
import pandas as pd 
import json

hist_df = pd.DataFrame(hist.history)

with open("sakla.json" ,"w") as f:
    hist_df.to_json(f) 
    
    
#%% load history
with open("sakla.json") as json_file:
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