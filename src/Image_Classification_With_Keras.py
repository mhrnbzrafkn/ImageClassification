import os
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model

# Define Values To Start

# Path To Datas
dataPath = './images/car/'
modelPath = dataPath + 'model.h5'

# Define data directories
train_dir = dataPath + 'train/'
test_dir = dataPath + 'validation/'

# Define Test Image Path
img_path = dataPath + 'test/img (3).jpg'

# Define the model
model = Sequential()
# Define image size and batch size
width, height = 88, 88
img_size = (width, height)
input_size = (width, height, 1)
batch_size = 32

# ----- ----- ----- ----- #

if (os.path.isfile(modelPath)):
    model = load_model(modelPath)
else:
    # Add convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_size))
    # Add max pooling layer with 2x2 pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the output of the previous layer
    model.add(Flatten())
    # Add a fully connected layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    # Add output layer with softmax activation for classification
    model.add(Dense(2, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ----- ----- ----- ----- #

    # Create an ImageDataGenerator object for data augmentation and normalization
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    # Create image generators for training and validation data
    X_train = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, color_mode='grayscale')
    X_test = val_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, color_mode='grayscale')

    # ----- ----- ----- ----- #

    # Train the model on the image generators
    model.fit(X_train, steps_per_epoch=X_train.n//batch_size, epochs=10, validation_data=X_test, validation_steps=X_test.n//batch_size)
    model.save(modelPath)

# ----- ----- ----- ----- #

# Load the image and resize it to match the input shape of the model
# img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img = cv.imread(img_path)
img = cv.resize(img, img_size)
img = cv. cvtColor(img, cv.COLOR_BGR2GRAY)
img = np.expand_dims(img, 2)
# Convert the image to a numpy array and normalize its pixel values
def normalizeImage(img):
    w = img.shape[0]
    h = img.shape[1]

    result = np.zeros((w, h, 1))

    for i in range(w):
        for j in range(h):
            result[i][j][0] = img[i][j][0] / 255
    
    return result

x = normalizeImage(img)
# Reshape the array to match the input shape of the model
x = np.expand_dims(x, axis=0)

# ----- ----- ----- ----- #

# Use the model to make a prediction on the image
prediction = model.predict(x)
# Get the class label with the highest probability
class_index = np.argmax(prediction)
# Print the predicted class label
print('Predicted class:', class_index)