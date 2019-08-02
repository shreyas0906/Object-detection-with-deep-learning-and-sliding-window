import os
from os import path
from random import shuffle

def checkForOnlyImages(fileLists):

    onlyImages = []

    for file in fileLists:
        if file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'):
            onlyImages.append(file)

    return onlyImages

def getAllTrainImages():

    trainFilesNames = []

    for trainDir, subDirs, fileLists in os.walk(os.path.join(os.getcwd(),'train')):
        onlyImages = checkForOnlyImages(fileLists)
        for file in onlyImages:
            trainFilesNames.append(os.getcwd() + '/' + os.path.join(dir, file))

    return shuffle(trainFilesNames)

def getAllTestImages():

    testFilesNames = []

    for testDir, subDirs, fileLists in os.walk(os.path.join(os.getcwd(), 'test')):
        onlyImages = checkForOnlyImages(fileLists)
        for file in onlyImages:
            testFilesNames.append(os.getcwd() + '/' + os.path.join(dir, file))

    return shuffle(testFilesNames)

def createImageList():


    if path.exists('train.txt'):
        os.remove(os.path.join(os.getcwd(), 'train.txt'))

    trainImagesList = getAllTrainImages()

    with open('train.txt', mode='w') as f:
        for item in trainImagesList:
            f.write(item)

    if path.exists('test.txt'):
        os.remove(os.path.join(os.getcwd(), 'test.txt'))

    testImagesList = getAllTestImages()

    with open('test.txt', mode='w') as f:
        for item in testImagesList:
            f.write(item)

def getLabels():

    for dir, subdir, _ in os.walk('train'):
        if len(subdir) != 0:
            return subdir

