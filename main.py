# importing the requied libraries
import time
import cv2
from keras.models import model_from_json # loading the model which was previously saved
import numpy as np
from skimage.measure import structural_similarity as ssim
import os

curr_dir = '/home/shreyas0906/PycharmProjects/project-swarm'
# Storing the predicted values
TP =[]
TN =[]
FP =[]
FN =[]
scp = []
scn = []
true_positive = cv2.imread('blended.jpeg', cv2.IMREAD_GRAYSCALE)
true_negative = cv2.imread('trn_174-roi.jpeg', cv2.IMREAD_GRAYSCALE)
tp = true_positive[:,:]/255
tn = true_negative[:,:]/255
#Test image to detect humans in the image
img = cv2.imread('trn_165.jpg', cv2.IMREAD_GRAYSCALE)
resized = img[65:200, 250:631]/255 # Pre-processing the image and normalize
wind_row, wind_col = 45,45 # dimensions of the image
img_rows, img_cols = 45,45
#loading the model first so that it can be tested in future cases
json_file = open('model.json', 'r')
loaded_model_from_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_from_json)
loaded_model.load_weights("model.h5")
print("Model is loaded from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Comparing the similarity between the predicted image and actual image
def compare_images(imageA, imageB):
    return ssim(imageA, imageB)

#generating the sliding window
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# predicting the image
def load_model(window_image):
    return loaded_model.predict(window_image)

#displaying the image on which the sliding window is displayed
def show_window():
    for(x,y, window) in sliding_window(resized, 15, (wind_row,wind_col)):
        if window.shape[0] != wind_row or window.shape[1] != wind_col:
            continue
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + wind_row, y + wind_col), (0, 255, 0), 2)

        t_img = resized[y:y+wind_row,x:x+wind_col]# the image which has to be predicted

        test_img = np.expand_dims(np.expand_dims(t_img,axis =0), axis=0) # expanding the dimensions of the image to meet the dimensions of the trained model
        prediction = load_model(test_img) # predict the image
        classes = prediction[0]

        if classes[0] > 0.99999: #Applying threshold
            print("human detected with a probability: ",classes[0], '\t', "x: ",x,'\t',"y: ", y)
            S = compare_images(t_img, tp)
            scp.append(S)
            if S > 0.5:
                TP.append(1)
            else: FP.append(1)
        else:
            print("background with a probability: ", 1-classes[1], '\t', "x: ",x,'\t',"y: ", y)
            S = compare_images(t_img, tn)
            scn.append(S)
            if S > 0.5:
                TN.append(1)
            else: FN.append(1)

        cv2.imshow("sliding_window", resized[y:y+wind_row,x:x+wind_col])
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.25)

def consfusion_matrix(TN,FN,TP,FP):
    print("The recall of the model: ", len(TP)/(len(TP)+len(FN)))
    print("The F1 score of the model", (2*len(TP))/(len(FP)+len(FN)+2*len(TP)))

show_window()
consfusion_matrix(TN,FN,TP,FP)
print('\t',"TN=",len(TN),'\t',"FP=",len(FP))
print('\t',"FN=",len(FN),'\t',"TP=",len(TP))
cv2.waitKey(0)
cv2.destroyAllWindows()

