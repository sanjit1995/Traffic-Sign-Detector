import pandas as pd
import os, random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from TrafficDetection.imageOperations import augmentDataInFiles
from distutils.dir_util import copy_tree, remove_tree
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch

img_path = "data/images/"

# Import labels.csv
labels_dataset = pd.read_csv("data/labels.csv")
temp_labels = labels_dataset.iloc[:, 0]

########### Uncomment below code if augmented data is lost ###########################
# Import images
i = 0
# max_count = 2010
# while i < 43:
#     file_count = 0
#     current_path = img_path + str(i) + "/"
#     # print(current_path)
#     for file in os.listdir(current_path):
#         img = cv.imread(os.path.join(current_path, file), 0)
#         # temp_img_list.append(img)
#         # temp_label_list.append(temp_train_labels[i])
#         file_count += 1
#     print(file_count)
#     augment_count = max_count - file_count
#     print(augment_count)
#     save_path = current_path + "new/"
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     if(augment_count < file_count):
#         new_files = 2
#     else:
#         new_files = round(augment_count / file_count)
#     print(new_files)
#     augmentDataInFiles(noOfNewFiles=new_files, fromDir=current_path, saveDir=save_path, maxFiles=augment_count)
#     copy_tree(save_path, current_path)
#     remove_tree(save_path)
#     i += 1

print("Image data adjustments done")

# Initialize the lists
label_count = 0
temp_img_list = []
temp_label_list = []

# Store the augmented images and labels
while i < 43:
    file_count = 0
    current_path = img_path + str(i) + "/"
    print(current_path)
    for file in os.listdir(current_path):
        img = cv.imread(os.path.join(current_path, file), 0)
        temp_img_list.append(img)
        temp_label_list.append(temp_labels[i])
        file_count += 1
    i += 1
    print(file_count)

# Convert the lists to numpy arrays
all_images = np.array(temp_img_list)
all_labels = np.array(temp_label_list)

# Create separate training and test data
train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2,
                                                                        shuffle=True)

# Normalize the image data
train_images = train_images / 255.0
test_images = test_images / 255.0
orig_test_images = test_images

# Reshape the arrays according to CNN input
train_images = train_images.reshape(len(train_images), 32, 32, 1)
test_images = test_images.reshape(len(test_images), 32, 32, 1)


##################  Training part  #######################
# For Hyper-parameter optimization
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=128, max_value=256, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(32, 32, 1)
    ))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
        activation='relu'
    ))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=64, max_value=256, step=32),
        activation='relu'
    ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(43, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='output',
    project_name='models'
)

# Search for the best model
tuner.search(train_images, train_labels, epochs=3, validation_split=0.1)

model = tuner.get_best_models(num_models=1)[0]

model.summary()

# Train the data with the best model
model.fit(train_images, train_labels, batch_size=100, epochs=10, validation_split=0.1, initial_epoch=3)

# Check the test data
m = 0
img = test_images[m]
img = np.expand_dims(img, axis=0)
prediction = model.predict_classes(img)
plt.imshow(orig_test_images[0])
plt.show()
print(prediction)
print("Original :")
print(test_labels[m])

#################### Just for printing things ##########################

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)
# plt.imshow(train_images[10])
# plt.show()
# cv.waitKey(0)
# print(train_labels[10])
