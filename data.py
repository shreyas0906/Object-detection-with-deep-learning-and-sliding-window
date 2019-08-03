#importing the required libraries for image processing
import cv2
import numpy as np
import os
from argparse import ArgumentParser
import utils

test_images = []
file_images = []
train_neg_images = []

#Specifying the paths to respective folders
def main(args):

    makeData(args.trainData)
    makeData(args.testData)


# TODO: write a script to rename all the images in a folder to a particular name. - completed
# TODO: write a script to append all the images to a list and assign labels accordingly.


def makeData(folder):

    imagesFiles = []
    imagesLabels = []

    for root, subDir, fileNames in os.walk(folder):


    #     if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png'):
    #         imagesFiles.append(f)
    #     if f.split('.')[0][0:3] == getLabel(args.trainData):
    #         imagesLabels.append(1)#Labeling the images accordi
    #     else:
    #         imagesFiles.append(0)# assigning the label 0 to the negative images.
    # trainImgs = np.ndarray((len(os.listdir(args.trainData)), 1, args.objectDim, args.objectDim)) / 255 # normalizing the image


def makeTestingImages():

    imagesFiles = []
    imagesLabels = []

    for f in os.listdir(args.testData):
        if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png'):
            trainImgs[i] = np.asarray(img)
            i+=1

    testLabels = np.asarray(train_img_label)# creating a numpy array for training labels

    for l in os.listdir(testing_path):
        if l.endswith('.jpeg') or l.endswith('.jpg') or l.endswith('.png'):
            train_neg_images.append(l)
        if l.split('.')[0][0:3] == 'trn':
            test_img_label.append(1)
        else:
            test_img_label.append(0)
    testImgs = np.ndarray((len(train_neg_images), 1, image_width, image_height))/255 # normalizing the image

    i = 0
    os.chdir(testing_path)
    for num in train_neg_images:
        if num.endswith('.jpeg') or num.endswith('.jpg') or num.endswith('.png'):
            neg_img = cv2.imread(num, cv2.IMREAD_GRAYSCALE)
            # ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
            testImgs[i] = np.asarray(neg_img)
            i+=1

    testLabels = np.asarray(test_img_label)

    return testLabels, testImgs, trainLabels, trainImgs

testLabels, testImgs, trainLabels, trainImgs = make_training_images()

#os.chdir(orig_path)

def saveData():

    print("Saving the testing images and labels into a npy file...")
    np.save("testLabels.npy",testLabels)
    np.save('testingImg.npy', testImgs)
    print("Saving the training images and labels into a npy file...")
    np.save("trainLabels.npy",trainLabels)
    np.save('trainImgs.npy', trainImgs)


if __name__ =='__main__':

        p = ArgumentParser()
        p.add_argument('trainDir', required=True, help='name of training data folder')
        p.add_argument('testDir', required=True, help='name of testing data folder')
        p.add_argument('objectDim', required=False, help='size of object to be detected. default=45', default=45)
        args = p.parse_args()
        
# Check the number of directories in training data. if more than two folders raise error.
# testing data. 
    






