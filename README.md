# Object-detection-with-deep-learning-and-sliding-window
Introduces an approach for object detection in an image with sliding window. 
The Idea behind this approach is to train a CNN model on images which we want to detect. After that make a sliding window on the image and query the trained model to produce appropriate output. 
The repository contains three files: make_data.py contains code to preprocess the image and append labels accordingly. In this case, if the file image starts with "trn" then it is assigned the label 1 else 0. The images are then normalized and appended into a numpy array. The numpy array is then saved as a .npy file for the model training. 
test-model-1.py file contains the code for the CNN model. The model and weights are saved into a .json file and h5 file format. 
sliding_window, the window dimensions are of 45 x 45, which can modified to your needs and the step size of the window is 12. The trained CNN model and weights are loaded. The image from the sliding window is the queried with the trained model. 
To produce the confusion matrix, true positive and true negative images are required. The sliding window image and the the true positive and true negative images are compared with the Structural Similarity index model. A threshold is set to differentiate between TN,FN,TP,FP. 

