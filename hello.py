from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import ImageDataGenerator,image 
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.models import Sequential,Model,load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
model = keras.models.load_model('Final_model_vgg.h5')

#define height and width of the image
height=300
width=300


base_model=VGG16(weights='imagenet',include_top=False,input_shape=(height,width,3))#define directory containing training and validation data)
    #define directory containing training and validation data
train_dir=r"C:\Users\NIVITHA\Desktop\Anti-Cyber-Bullying-master\dataset\train"
validation_dir=r"C:\Users\NIVITHA\Desktop\Anti-Cyber-Bullying-master\dataset\val"

    #number of batches the data has to be divided into
batch_size=32

    #create datagen and generator to load the data from training directory
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=90,horizontal_flip=True,vertical_flip=True)
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(height,width),batch_size=batch_size)

    #create datagen and generator to load the data from validation directory
validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=90,horizontal_flip=True,vertical_flip=True)
validation_generator=validation_datagen.flow_from_directory(validation_dir,target_size=(height,width),batch_size=batch_size)
    #our own model which will be added onto the VGG16 model
def build_finetune_model(base_model,dropout,fc_layers,num_classes):
    for layer in base_model.layers:
        layer.trainable=False

        x=base_model.output
        x=Flatten()(x)
        for fc in fc_layers:
            x=Dense(fc,activation='relu')(x)
            x=Dropout(dropout)(x)
        
        predictions=Dense(num_classes,activation='softmax')(x)

        finetune_model=Model(inputs=base_model.input,outputs=predictions) 
        
        return finetune_model

class_list=['safe','unsafe'] #the labels of our data
FC_Layers=[1024,1024]
dropout=0.5

finetune_model=build_finetune_model(base_model,dropout=dropout,fc_layers=FC_Layers,num_classes=len(class_list))

if(1):
    #define number of epochs(the number of times the model will be trained) and number of training images
    num_epochs=20
    num_train_images=692

    #checkpoint in case anything goes wrong
    checkpoint=ModelCheckpoint("Final_model_vgg.h5",monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=False,mode='auto',period=1)
    early=EarlyStopping(monitor='val_acc',min_delta=0,patience=40,verbose=1,mode="auto")

    #compile the model before using
    finetune_model.compile(loss="categorical_crossentropy",optimizer=optimizers.SGD(lr=0.000001,momentum=0.9),metrics=['accuracy'])

    #train the model
    k=finetune_model.fit_generator(generator=train_generator,steps_per_epoch=num_train_images//batch_size,epochs=num_epochs,validation_data=validation_generator,validation_steps=1,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

    #save the model
    finetune_model.save(r"C:\Users\NIVITHA\Desktop\Anti-Cyber-Bullying-master\Final_model_vgg.h5")
if (0):
    
    #testing the model
    from keras.preprocessing import image
    import matplotlib.image as mpimg
    img = image.load_img(r"C:\Users\NIVITHA\Desktop\Anti-Cyber-Bullying-master\dataset\test\5.jpg",target_size=(300,300))
    img = np.asarray(img)
    plt.imshow(mpimg.imread(r"C:\Users\NIVITHA\Desktop\Anti-Cyber-Bullying-master\dataset\test\5.jpg"))
    plt.ion()
    plt.show()
    img = np.expand_dims(img, axis=0)
    #finetune_model.load_weights(r"C:\Users\NIVITHA\Desktop\Anti-Cyber-Bullying-master\Final_model_vgg.h5")

    output=finetune_model.predict(img) #predicting the image using model created
    if(output[0][0]>output[0][1]): #comparison
        print("safe")
    else:
        print("unsafe")
