import cv2  # for capturing videos
import math  # for mathematical operations
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def augmentData(img_file, noOfNewFiles, saveDir, imageName):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(img_file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=saveDir, save_prefix=imageName, save_format='jpg'):
        i += 1
        if i >= noOfNewFiles:
            break


def resizeImages():
    count = 0
    for filename in os.listdir("data/Tom and Jerry/Tom/p/"):
        # print(filename)
        img = cv2.imread(os.path.join("data/Tom and Jerry/Tom/p/", filename), 0)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite("data/Tom and Jerry/Tom/p/" + str(count) + "_positive.jpg", img)
        count += 1


# resizeImages()
def storeRelativePaths():
    f = open("../opencv-haar-classifier-training/positives.txt", "w")
    for filename in os.listdir("../opencv-haar-classifier-training/positive_images"):
        if str(filename) == ".gitkeep":
            continue
        # print(filename)
        f.write("positive_images/" + str(filename) + "\n")


# storeRelativePaths()

# augmentData(img_file="data/train-images/frame0.jpg", noOfNewFiles=9, saveDir="frame0",imageName="frame0")

def augmentDataInFiles(noOfNewFiles, fromDir, saveDir, maxFiles):
    count = 0
    for filename in os.listdir(fromDir):
        if filename == "new":
            continue
        if count >= maxFiles:
            return
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest')

        img = load_img(str(fromDir) + str(filename))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=saveDir, save_prefix=count, save_format='png'):
            i += 1
            if i >= noOfNewFiles:
                break
        count += noOfNewFiles


# augmentDataInFiles(noOfNewFiles=2, fromDir="data/Tom and Jerry/Tom/p/", saveDir="data/Tom and Jerry/Tom/p/new/")
