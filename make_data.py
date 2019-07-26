#importing the required libraries for image processing
import cv2
import numpy as np
import os

test_images = []
file_images = []
train_neg_images = []
#Specifying the paths to respective folders
train_pos_path = '/home/shreyas0906/PycharmProjects/project-swarm/data-1/train/pos'
train_neg_path = '/home/shreyas0906/PycharmProjects/project-swarm/data-1/train/neg'
testing_path = '/home/shreyas0906/PycharmProjects/project-swarm/data-1/test/'
orig_path = '/home/shreyas0906/PycharmProjects/project-swarm/'
image_width, image_height = 45,45 # specifying the image dimensions
train_img_label = []
testing_img_label = []
test_img_label = []

def make_training_images():
    i = 0
    for f in os.listdir(train_pos_path):
        if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png'):
            file_images.append(f)
        if f.split('.')[0][0:3] == 'trn': # assigning the label 1 to the cropped images. Note that the file name differentiates between
            train_img_label.append(1)#the positive and negative images
        else:
            train_img_label.append(0)# assigning the label 0 to the negative images.
    training_imgs = np.ndarray((len(file_images), 1, image_width, image_height))/255 # normalizing the image

    os.chdir(train_pos_path)
    for file in file_images:
        if file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
            training_imgs[i] = np.asarray(img)
            i+=1

    training_img_label = np.asarray(train_img_label)# creating a numpy array for training labels

    for l in os.listdir(testing_path):
        if l.endswith('.jpeg') or l.endswith('.jpg') or l.endswith('.png'):
            train_neg_images.append(l)
        if l.split('.')[0][0:3] == 'trn':
            test_img_label.append(1)
        else:
            test_img_label.append(0)
    testing_imgs = np.ndarray((len(train_neg_images), 1, image_width, image_height))/255 # normalizing the image

    i = 0
    os.chdir(testing_path)
    for num in train_neg_images:
        if num.endswith('.jpeg') or num.endswith('.jpg') or num.endswith('.png'):
            neg_img = cv2.imread(num, cv2.IMREAD_GRAYSCALE)
            # ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
            testing_imgs[i] = np.asarray(neg_img)
            i+=1

    testing_img_label = np.asarray(test_img_label)

    return testing_img_label, testing_imgs, training_img_label, training_imgs

testing_img_label,testing_imgs,training_img_label,training_imgs = make_training_images()

os.chdir(orig_path)
print("The shape of tesing images: ",np.shape(testing_imgs),'\n',
      "The shape of tesing lables: ", np.shape(testing_img_label))
print( "The shape of training images ",np.shape(training_imgs),'\n',
       "The shape of training labels: ", np.shape(training_img_label))

if "testing_img_label.npy" in os.listdir(os.getcwd()):
    print("Saving the testing images into a npy file...")
    print("Saving testing image labels...")
    np.save("testing_img_label.npy",testing_img_label)
    np.save('testing_img.npy', testing_imgs)
    print("Saving the training images into a npy file...")
    print("Saving image labels...")
    np.save("training_img_labels.npy",training_img_label)
    np.save('training_img.npy', training_imgs)






