from flask import Flask, render_template,request
from flask import Flask,render_template,url_for,request
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import time
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
import matplotlib.image as mpimg
from keras.models import load_model
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg
import cv2
from PIL import Image
import numpy as np
from skimage import transform
import pickle
import joblib
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
model = keras.models.load_model('Final_model_vgg.h5')
height=300
width=300

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import pickle
import RNN


app = Flask(__name__)

@app.route('/')


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (300, 300, 3))
    np_image = np.expand_dims(np_image, axis=0)
    img=mpimg.imread(filename)
    plt.imshow(img)
    return np_image


start = time.time()
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", name="Tariq")



@app.route('/predict',methods=['POST'])
def predict():


        file=open("model.pickle","rb")
        model1=pickle.load(file)
        file.close()
        print('hi')
        max_words = 1000
        max_len = 100
        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            print(data)
            tok = pickle.load(open("tok.pickle","rb"))
            test_sequences=""
            test_sequences = tok.texts_to_sequences(data)
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
                lb='hate speech'
            else:
                print("normal")
                lb='normal'
        return render_template("prediction.html", data=lb)

    
@app.route('/prediction', methods=["POST"])



def prediction():
    img = request.files['img']
    img.save('./test.jpg')
    
    img = image.load_img("test.jpg",target_size=(300,300))
    img = np.asarray(img)
    plt.imshow(mpimg.imread("test.jpg"))
    plt.ion()
    plt.show()
    img = np.expand_dims(img, axis=0)
    #finetune_model.load_weights(r"C:\Users\NIVITHA\Desktop\Anti-Cyber-Bullying-master\Final_model_vgg.h5")

    output=model.predict(img) #predicting the image using model created
    if(output[0][0]>output[0][1]): #comparison
        print("safe")
        lbl='safe'
    else:
        print("Unsafe")
        lbl='UnSafe'
    print("Overall the pic is identified as :" + lbl)


                

    return render_template("prediction.html", data=lbl)


if __name__ =="__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run("127.0.0.1", port=5000, debug=False,threaded=False)
