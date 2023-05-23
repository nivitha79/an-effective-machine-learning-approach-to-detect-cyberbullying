#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from keras.utils import pad_sequences
from keras.preprocessing.sequence import pad_sequences
#from keras.optimizers import RMSprop
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import pickle
import RNN

if(1):
    file=open("model.pickle","rb")
    model1=pickle.load(file)
    file.close()
if(0):
    df = pd.read_csv("train.csv")
    dft=pd.read_csv("test_with_solutions.csv")
    df.head()
    dft.head()
    df.drop(['Date'],axis=1,inplace=True)
    dft.drop(['Date','Usage'],axis=1,inplace=True)
    
    sns.countplot(df.Insult)
    plt.xlabel('Label')
    plt.title('Number of non-bully vs bully messages in trianing dataset')
    plt.show()
    
    X = df.Comment
    Y = df.Insult
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1,1)
    le = LabelEncoder()
    
    #X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
    X_train,Y_train=X,Y
    max_words = 1000
    max_len = 100
    tok1 = Tokenizer(num_words=max_words)
    tok1.fit_on_texts(X_train)
    fl=open("tok1.pickle","wb")
    pickle.dump(tok1,fl)
    fl.close()
    sequences = tok1.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    
    model1 = RNN.RNN()
    model1.summary()
    model1.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    
    k=model1.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
              validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    f=open("model1.pickle","wb")
    pickle.dump(model1,f)
    f.close()
    # loss
    plt.plot(k.history['loss'],label="train loss")
    plt.plot(k.history['val_loss'],label="val loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("LossVal_loss")
    # accuracy
    plt.plot(k.history['accuracy'],label="train accuracy")
    plt.plot(k.history['val_accuracy'],label="val accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("AccVal_accuracy")
max_words = 1000
max_len = 100
X_test=[input("Enter ")]
print(X_test)
tok = pickle.load(open("tok.pickle","rb"))
test_sequences=""
test_sequences = tok.texts_to_sequences(X_test)
print(test_sequences)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
print(test_sequences_matrix)

ans=model1.predict(test_sequences_matrix,batch_size=None,verbose=0,steps=None)
print(ans)
st=""
st=st.join(str(ans[0][0]))
print(st)
if float(st)>0.5:
    print("hate speech")
else:
    print("normal")
