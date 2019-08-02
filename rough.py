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
        checkForOnlyImages(fileLists)
        trainFilesNames.append(os.path.join(trainDir, ))

    return shuffle(trainFilesNames)


for dir, subdir, filelists in os.walk('root'):

    # onlyImages = checkForOnlyImages(filelists)
    print(subdir)
    # for file in onlyImages:
    #     print(os.getcwd()+'/'+ os.path.join(dir,file))

