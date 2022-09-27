import matplotlib.pyplot as plt
import numpy as np
import keras
import cv2

from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
from keras import layers



#%% Veri Seti Ayarlama
egitimPath = ""
testPath = ""

inputShape=(224,224,3)
egitimDataGenerator = ImageDataGenerator(rescale=1./255,
                                horizontal_flip =True,
                                vertical_flip =True,
                                rotation_range=45,
                                height_shift_range=0.2,
                                width_shift_range=0.2,
                                fill_mode="nearest")
testDataGenerator = ImageDataGenerator(rescale=1./255)

egitimGenerator = egitimDataGenerator.flow_from_directory(egitimPath,
                                                        target_size=(224,224),
                                                        class_mode= "categorical",
                                                        batch_size=64)
                                                        
testGenerator = testDataGenerator.flow_from_directory(testPath,
                                                        target_size=(224,224),
                                                        class_mode= "categorical",
                                                        batch_size=16)



#%% Model Oluşturma
#modeli fonk içine kurup fonksiyonu çağırma
def yanginTespiti(inputShape):
    model= keras.models.Sequential([
                                    layers.Conv2D(96,(11,11),strides=(4,4),activation="relu",input_shape=inputShape),
                                    layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
                                    
                                    layers.Conv2D(256,(5,5),activation="relu"),
                                    layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
                                    
                                    layers.Conv2D(512,(5,5),activation="relu"),
                                    layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
                                    
                                    layers.Flatten(),
                                    layers.Dropout(0.3),
                                    
                                    layers.Dense(2048, activation="relu"),
                                    layers.Dropout(0.3),
                                    
                                    layers.Dense(1024, activation="relu"),
                                    layers.Dropout(0.3),

                                    layers.Dense(2, activation="softmax")
                                    ])
    
    #compile etme
    model.compile(loss = "categorical_crossentropy",
                optimizer= Adam(lr=1e-4),                
                metrics=["acc"])
    return model



model = yanginTespiti(inputShape)
model.summary()

grafikler = model.fit(egitimGenerator,
                    steps_per_epoch=15,
                    epochs=50,
                    validation_data=testGenerator,
                    validation_steps=15
                    )

#%% görselleştirme
acc = grafikler.history["acc"]
val_acc = grafikler.history["val_acc"]

loss = grafikler.history["loss"]
val_loss = grafikler.history["val_loss"]

plt.figure()
epochs = range(0,50)

plt.plot(epochs,acc,"g",label="egitim accuracy")
plt.plot(epochs,val_acc,"black",label="test accuracy")
plt.title("egitim-test accuracy")
plt.legend(loc=0)
plt.figure()
plt.show()

plt.plot(epochs,loss,"r",label="egitim accuracy")
plt.plot(epochs,val_loss,"blue",label="test accuracy")
plt.title("egitim-test loss")
plt.legend(loc=0)
plt.figure()
plt.show()

model.save("yanginTespiti.h5")



#%% Deneme, Test
model = load_model("yanginTespiti.h5")
videoPath = "  "
resimPath = "  "



#Resimden Tespit
orjinal = cv2.imread(resimPath)
img = np.asarray(orjinal)
img = cv2.resize(img, (224,224))
img = img/255
img = img.reshape(1,224,224,3)

tahmin = model.predict(img)
pred = np.argmax(tahmin[0])

dogruluk = tahmin[0][pred]
dogrulukCikti = "% {:.2f}".format(dogruluk*100)

if pred == 1:
    label = "Fire"
else:
    label = "Neutral"
    
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0,255,0)

cv2.putText(orjinal, label, (35,60), font, 1, color, 2)
cv2.putText(orjinal, dogrulukCikti, (35,100), font, 1, color, 2)

cv2.imshow("tahmin", orjinal)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Videodan Tespit
video = cv2.VideoCapture(videoPath)
while True:
    ret,frame = video.read()
    
    img = np.asarray(frame)
    img = cv2.resize(img, (224,224))

    img = img/255
    img = img.reshape(1,224,224,3)

    tahmin = model.predict(img)
    pred = np.argmax(tahmin[0])

    dogruluk = tahmin[0][pred]
    dogrulukCikti = "% {:.2f}".format(dogruluk*100)

    if pred == 1:
        label = "Fire"
    else:
        label = "Neutral"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,255,0)

    cv2.putText(frame, label, (35,60), font, 1, color, 2)
    cv2.putText(frame, dogrulukCikti, (35,100), font, 1, color, 2)
    cv2.imshow("tahmin", frame)
    
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break

video.release()
cv2.destroyAllWindows()