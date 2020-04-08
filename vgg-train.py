from keras.layers import Flatten,Dense,Input,GlobalAveragePooling2D,GlobalMaxPooling2D,Activation,Conv2D,MaxPooling2D,BatchNormalization,AveragePooling2D,Dropout
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# the main VGG16 network as implemented in the paper
def VGG16(input_shape=None, include_top=True, pooling=None, classes=2622,input_tensor=None):
    input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=48,data_format=K.image_data_format(),require_flatten=True)

    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(4096, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dense(classes, name='fc8')(x)
        x = Activation('softmax', name='fc8/softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    weights_path = 'rcmalli_vggface_tf_vgg16.h5'

    model = Model(img_input, x, name='vggface_vgg16')
    model.load_weights(weights_path, by_name=True)

    return model

# function to return the train/test data splits and ready to train and test the network
def get_train_test_data():
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    class_labels=[]

    person_folders = os.listdir('dataset/')
    for i,person in enumerate(person_folders):
        class_labels.append(person)
        types=['train','test']
        for t in types:
            image_names = os.listdir('dataset/{}/{}'.format(person,t))
            for image_name in image_names:
                img=image.load_img('dataset/{}/{}/{}'.format(person,t,image_name),target_size=(224,224))
                img=image.img_to_array(img)
                img=np.expand_dims(img,axis=0)
                img=preprocess_input(img)
                if(t=='train'):
                    x_train.append(img)
                    y_train.append(i)
                else:
                    x_test.append(img)
                    y_test.append(i)
    
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)

    trainX = np.reshape(x_train, (x_train.shape[0], 224, 224, 3))/255
    trainY = np_utils.to_categorical(y_train,len(class_labels))

    testX = np.reshape(x_test, (x_test.shape[0],224,224,3))/255
    testY = np_utils.to_categorical(y_test,len(class_labels))

    return (trainX,trainY,testX,testY,class_labels)

# function to plot the confusion matrix
def plot_confusion_matrix(cm, index, classes, title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title+' for model {}'.format(index))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    
    #plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion matrix for model {}'.format(index))
    plt.clf()

# function that returns one of 5 implemented models (in the classification block only)
def get_model(test_num):
    vgg_model = VGG16(input_shape=(224, 224, 3), include_top=False,pooling='max')
    x = vgg_model.get_layer('pool5').output

    # different classifiers according to test number
    if(test_num==0):
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(4096, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dense(num_classes, name='fc8')(x)
        x = Activation('softmax', name='fc8/softmax')(x)
    elif(test_num==1):
        x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(128,activation='relu')(x)
        x = Dense(num_classes,activation='softmax')(x)
    elif(test_num==2):
        x = Conv2D(10, (3,3), activation='relu',padding='same')(x)
        x = Conv2D(10, (3,3), activation='relu',padding='same')(x)
        x = Flatten()(x)
        x = Dense(10, activation='relu')(x)
        x = Dense(num_classes,activation='softmax')(x)
    elif(test_num==3):
        x = Conv2D(10, (5,5), activation='relu',padding='same')(x)
        x = Conv2D(10, (3,3), activation='relu',padding='same')(x)
        x = Flatten()(x)
        x = Dense(num_classes,activation='softmax')(x)
    elif(test_num==4):
        x = Flatten(name='flatten')(x)
        x = Dense(10, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=vgg_model.inputs,outputs=x)

if __name__=='__main__':

    # get the train/test data
    trainX,trainY,testX,testY,class_labels = get_train_test_data()
    num_classes = len(class_labels)

    # iterating over the 5 implemented models
    # each is just the main VGG-Face model with the classification block changed
    num_epochs=[15,15,25,15,15]
    for index in range(5):
        print("Starting with model {}".format(index))
        model=get_model(index)

        # set the first 5 blocks (consisting of 18 layers) to non-trainable to be able to 
        # train the classification block only
        for i in range(18):
            model.layers[i].trainable = False

        # compile and train the model
        model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
        history = model.fit(trainX,trainY,validation_data=(testX,testY),epochs=num_epochs[index])
        
        # save model weights file to disk
        model_json = model.to_json()
        with open("model"+str(index)+".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model"+str(index)+".h5")
        print("Model "+ str(index) +" saved to disk")

        # get history of training for plotting accuracy vs num_of_epochs
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')

        # save the plots
        plt.savefig('results/accuracy graph for model {}'.format(index))
        plt.clf()
        print("Saved plot {} to disk".format(index))

        # write the confusion matrix results to file
        predY = model.predict(testX)
        rounded_labels = np.argmax(testY,axis=1)
        rounded_pred = np.argmax(predY,axis=1)
        cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_pred)
        plot_confusion_matrix(cm=cm, classes=class_labels, title='Confusion Matrix',index=index)
        print("Saved confusion matrix {} to disk".format(index))


        # write the accuracy results to file
        score,acc = model.evaluate(testX,testY)
        with open("results/results_exp.txt", "a") as myfile:
            myfile.write('Test accuracy for model {}: {}\n\n'.format(index,acc))

    print("Done")
