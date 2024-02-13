# Image Classification
> In this repository, I tried to use [`Keras`](https://keras.io/) library and implement a simple project to classify images.

## Introduction
Welcome to the `ImageClassification` repository! This Python script defines and trains a convolutional neural network using `Keras` to classify `images` of `cars` and `non-cars`, with `data augmentation` and `normalization`, and provides a `prediction` for a `test image`.

Whether you are interested in the fundamentals of `image classification` or seeking `hands-on exploration`, this repository provides an opportunity to delve into `image classification`. To get started, follow the `usage instructions` and explore the `script` located in the `./src` folder. Happy coding!

## Usage
Simply clone the repository and execute the Python file inside the `./src` folder to utilize it. If you wish to replace the images or add another class of images, all you need to do is move your images into the `./src/images` folder and update the file paths. To run the code, execute the following commands:
```bash
git clone https://github.com/mhrnbzrafkn/ImageClassification.git
cd ImageClassification
pip install -r requirements.txt
python Image_Classification_With_Keras.py
```

## Project Structure
-   `./src/Image_Classification_With_Keras.py`: Main Python file; you should run this file.
    
-   `./src/images`: Main images folder that contains all images you want to classify.
    
    Here is some explanation of subfolders:
    
    -   `./src/images/car/train`: This folder contains two subfolders named `/0` and `/1`, representing the classes in our project. Class `0` includes images that do not depict cars, while class `1` comprises images depicting cars.
        
    -   `./src/images/car/validation`: Similarly, this folder contains two subfolders named `/0` and `/1`, mirroring the classes in the `/train` folder.
        
    -   `./src/images/car/test`: This folder contains a selection of images intended for classification after training the model.

## Demo
The output during training will resemble the following:
```
Found 240 images belonging to 2 classes.
Found 60 images belonging to 2 classes.
Epoch 1/10
7/7 [==============================] - 1s 112ms/step - loss: 2.5030 - accuracy: 0.4663 - val_loss: 0.8156 - val_accuracy: 0.4688
Epoch 2/10
7/7 [==============================] - 1s 88ms/step - loss: 1.1831 - accuracy: 0.5096 - val_loss: 0.5073 - val_accuracy: 0.6250
Epoch 3/10
7/7 [==============================] - 1s 88ms/step - loss: 0.5856 - accuracy: 0.6731 - val_loss: 0.3743 - val_accuracy: 0.9062
Epoch 4/10
7/7 [==============================] - 1s 89ms/step - loss: 0.4078 - accuracy: 0.8413 - val_loss: 0.2537 - val_accuracy: 0.9375
Epoch 5/10
7/7 [==============================] - 1s 93ms/step - loss: 0.3242 - accuracy: 0.8558 - val_loss: 0.2341 - val_accuracy: 1.0000
Epoch 6/10
7/7 [==============================] - 1s 88ms/step - loss: 0.2405 - accuracy: 0.9952 - val_loss: 0.2005 - val_accuracy: 0.9688
Epoch 7/10
7/7 [==============================] - 1s 86ms/step - loss: 0.1908 - accuracy: 0.9808 - val_loss: 0.1187 - val_accuracy: 1.0000
Epoch 8/10
7/7 [==============================] - 1s 90ms/step - loss: 0.1596 - accuracy: 0.9808 - val_loss: 0.1384 - val_accuracy: 0.9688
Epoch 9/10
7/7 [==============================] - 1s 89ms/step - loss: 0.1219 - accuracy: 0.9952 - val_loss: 0.0616 - val_accuracy: 1.0000
Epoch 10/10
7/7 [==============================] - 1s 92ms/step - loss: 0.0956 - accuracy: 0.9904 - val_loss: 0.1175 - val_accuracy: 0.9688
1/1 [==============================] - 0s 63ms/step
Predicted class: 1
```
This output shows the number of images found for training and validation, followed by the training progress over 10 epochs, and finally, the prediction result.

## Dependencies
- [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
- [![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
- [![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
- [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

## Contributing
We welcome contributions! Please [Contact Me](https://www.linkedin.com/in/mehran-bazrafkan/) before making a pull request or raising issues.
