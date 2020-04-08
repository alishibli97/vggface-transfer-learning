from keras.models import model_from_json
import numpy as np
import json
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

json_file = open('model0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model0.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])

def decode_prediction(pred):
    x = np.where(pred == 1)[1][0]
    if(x==0): return 'Andrew Cuomo'
    elif(x==1): return 'Anothy Faici'
    elif(x==2): return 'Bill Gates'
    elif(x==3): return 'Dolores Abernathy'
    elif(x==4): return 'Emilia Clarke'
    elif(x==5): return 'Fadlo Khouri'
    elif(x==6): return 'Hamad Hassan'
    elif(x==7): return 'Marcel Ghanem'
    elif(x==8): return 'Ali Shibli'
    elif(x==9): return 'Tedros Adhanom'
    elif(x==10): return 'Donald Trump'
    elif(x==11): return 'Ali Khayat'

img=image.load_img('me.jpg',target_size=(224,224))
img = np.expand_dims(img,axis=0)
img = preprocess_input(img)
preds = loaded_model.predict(img)

print("This is {}".format(decode_prediction(preds)))