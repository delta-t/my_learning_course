# import the necessary packages
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import build_montages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2


class MiniVGGNet:
    @staticmethod
    def build(height, width, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        # first Conv -> ReLU -> Conv -> ReLU -> Pool layer set
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second Conv -> ReLU -> Conv -> ReLU -> Pool layer set
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC -> ReLU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model


# initialize the number of epochs to train for, base learning rate,
# and batch size
epochs = 25
learn_rate = 1e-2
batch_size = 32
# grab the Fashion MNIST dataset (if this is your first time running
# this the dataset will be automatically downloaded)
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
#   num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
    testX = testX.reshape((testX.shape[0], 1, 28, 28))

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# one-hot encode the training and testing labels
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# initialize the label names
label_names = ['top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=learn_rate, momentum=0.9, decay=learn_rate / epochs)
model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# train the network
print("[INFO] training model")
H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              batch_size=batch_size, epochs=epochs)

# make predictions on the test set
preds = model.predict(testX)

# show a nicely formatted classification report
print('[INFO] evaluating network...')
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
                            target_names=label_names))

# plot the training loss and accuracy
N = epochs
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc='lower left')
plt.savefig("plot.png")

# initialize our list of output images
images = []

# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
    # classify the clothing
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = label_names[prediction[0]]

    # extract the image from the testData if using 'channels_first'
    # ordering
    if K.image_data_format() == 'channels_first':
        image = (testX[i][0] * 255).astype("uint8")

    # otherwise, we are using "channels_last" ordering
    else:
        image = (testX[i] * 255).astype("uint8")

    # initialize the text label color as green(correct)
    color = (0, 255, 0)

    # otherwise, the class label prediction is incorrect
    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)

    # merge the channels into one image and resize the image from
    # 28x28 to 96x96 so we can better see it and then draw the
    # predicted label on the image
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)

    # add the image to our list of output images
    images.append(image)

# construct the montage for the images
montage = build_montages(images, (96, 96), (4, 4))[0]

# show the output montage
cv2.imshow('Fashion MNIST', montage)
cv2.waitKey(0)
