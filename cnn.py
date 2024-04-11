# lets implement a simple CNN Model with the following architecture:
# Convolutional Layer with 32 filters and kernel size of 3x3, and RLU activation function
# Max Pooling Layer with pool size of 2x2
# Convolutional Layer with 64 filters and kernel size of 3x3, and RLU activation function
# Global Average Pooling Layer
# Dense Layer with 5 units and softmax activation function

# we will then train the model using the train folder in the current directory
# the train folder has 5 subfolders, each representing a class
# butterfly, cow, elephant, sheep, and squirrel
# and evaluate the model using the test folder in the current directory

# lets get started 

# import the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np

# # define the model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     GlobalAveragePooling2D(),
#     Dense(5, activation='softmax')
# ])

# # compile the model
# model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
# model.summary()

# # create the ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# # create the train and test generators
# train_generator = train_datagen.flow_from_directory('train', target_size=(128, 128), batch_size=32, class_mode='sparse')
# test_generator = test_datagen.flow_from_directory('test', target_size=(128, 128), batch_size=32, class_mode='sparse')

# # train the model
# model.fit(train_generator, epochs=10)

# # evaluate the model
# model.evaluate(test_generator)

# # save the model
# model.save('cnn_model.h5')

# # lets now make a prediction using the model

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# # load the model
# model = load_model('cnn_model.h5')

# # load an image
# img = image.load_img('test/sheep/OIP-ZxOzZFPi6qVZd_vINRU5gwHaFj.jpeg', target_size=(128, 128))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)

# # make a prediction
# prediction = model.predict(img_array)
# print(prediction)

# # get the class with the highest probability

# class_index = np.argmax(prediction)
# print(class_index)

# ok now lets swtich to a tranfer learning approach
# we will use the MobileNetV2 model as the base model
# and add a GlobalAveragePooling2D and Dense layer to the model

# # lets get started

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# # load the base model
# base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# # add the GlobalAveragePooling2D and Dense layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(5, activation='softmax')(x)

# # create the model
# model = Model(inputs=base_model.input, outputs=x)

# # compile the model
# model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
# model.summary()

# # create the ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# # create the train and test generators
# train_generator = train_datagen.flow_from_directory('train', target_size=(128, 128), batch_size=32, class_mode='sparse')
# test_generator = test_datagen.flow_from_directory('test', target_size=(128, 128), batch_size=32, class_mode='sparse')

# # train the model
# model.fit(train_generator, epochs=10)

# # evaluate the model
# model.evaluate(test_generator)

# # save the model
# model.save('transfer_learning_model.h5')

# # lets do another approach with data augmentation this time
# # we will use the MobileNetV2 model as the base model again

# # # lets get started

# # load the base model
# base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# # add the GlobalAveragePooling2D and Dense layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(5, activation='softmax')(x)

# # create the model
# model = Model(inputs=base_model.input, outputs=x)

# # compile the model
# model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
# model.summary()

# # create the ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# test_datagen = ImageDataGenerator(rescale=1./255)

# # the data augmentation is happening in the train_datagen ImageDataGenerator
# # by setting the rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip, and fill_mode parameters

# # create the train and test generators
# train_generator = train_datagen.flow_from_directory('train', target_size=(128, 128), batch_size=32, class_mode='sparse')
# test_generator = test_datagen.flow_from_directory('test', target_size=(128, 128), batch_size=32, class_mode='sparse')

# # train the model
# model.fit(train_generator, epochs=10)

# # evaluate the model
# model.evaluate(test_generator)

# # save the model
# model.save('transfer_learning_data_augmentation_model.h5')

# lets now do the same but with a different base model 
# we will use the InceptionV3 model as the base model
# then we will test with the accuracy metric

# # lets get started

# from tensorflow.keras.applications import InceptionV3

# # load the base model
# base_model = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# # add the GlobalAveragePooling2D and Dense layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(5, activation='softmax')(x)

# # create the model
# model = Model(inputs=base_model.input, outputs=x)

# # compile the model
# model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
# model.summary()

# # create the ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# test_datagen = ImageDataGenerator(rescale=1./255)

# # create the train and test generators
# train_generator = train_datagen.flow_from_directory('train', target_size=(128, 128), batch_size=32, class_mode='sparse')
# test_generator = test_datagen.flow_from_directory('test', target_size=(128, 128), batch_size=32, class_mode='sparse')

# # train the model
# model.fit(train_generator, epochs=10)

# # evaluate the model
# model.evaluate(test_generator)

# # save the model
# model.save('transfer_learning_data_augmentation_model_inceptionv3.h5')

# now lets do one final mdel with a base that we havent used before
# we will use the ResNet50 model as the base model
# then we will test with the accuracy metric

# lets get started

from tensorflow.keras.applications import ResNet50

# load the base model
base_model = ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# add the GlobalAveragePooling2D and Dense layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(5, activation='softmax')(x)

# create the model
model = Model(inputs=base_model.input, outputs=x)

# compile the model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

# create the ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

# create the train and test generators
train_generator = train_datagen.flow_from_directory('train', target_size=(128, 128), batch_size=32, class_mode='sparse')
test_generator = test_datagen.flow_from_directory('test', target_size=(128, 128), batch_size=32, class_mode='sparse')

# train the model
model.fit(train_generator, epochs=10)

# evaluate the model
model.evaluate(test_generator)

# save the model
model.save('transfer_learning_data_augmentation_model_resnet50.h5')

