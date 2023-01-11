import numpy as np
import cv2
import time
import os
import joblib
import sys
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from pqdict import pqdict
'''
The find phone program use to predict the phone location for a give picture (or a folder),
using HOG, SVM model, and sliding window strategy

Author: Shusen Lin
Date: 1/9/2023
'''
SCALE_PERCENTAGE = 100 # scale of original img, aim to speed up the computation
TRUST_THRESHOLD = 0.5  # potential bounding box trust threshold
ACCURACY = False   #True to enable the model accuracy test
VISUALIZE = False  #True to enable the bounding box visualization
SINGLE_TEST = True #True to enable the single input test

def tic():
  return time.time()

def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

class PhoneDetector(): 
    def __init__(self,folder,model):
        self.model = model
        self.folder = folder
        
    def sliding_window(self,img,window_size=(44,44),step_size=(5,5)): 
        '''
        The function for yield the sliding window for a given img

        Inputs: 1.img: 490x326 pixels original image (BGR format)
                2.window_size: sliding window size in pixels, same as training window size, default (44,44)
                3.step_size: sliding window step size, default x = y = 5 pixels(5,5)

        Output: yield sliding window up-left corner position and window image
        '''
        for y in range(0, img.shape[0], step_size[1]):
            for x in range(0, img.shape[1], step_size[0]):
                yield (x, y, img[y: y + window_size[1], x: x + window_size[0]])

    def find_phone_box(self, img ,window_size=(44,44),step_size=(5,5),visualize=False):
        '''
        The main function to detect the phone position using trained model, pyramid feature is disabled 
        for this approach.

        Inputs: 1.img: 490x326 pixels original image (BGR format)
                2.window_size: sliding window size in pixels, same as training window size, default (44,44)
                3.step_size: sliding window step size, default x = y = 5 pixels(5,5)
                4.visualize: True to enable the box visualization of phone detection, default False
               
        Output: 1.phone_position: center position of the bounding box in percentage
		'''
        test_img = img
        height,width  = test_img.shape[0:2] 
        possible_boxes = pqdict({}, reverse=True)
        # pyramid_imgs = tuple(pyramid_gaussian(test_img, downscale=1.5, channel_axis=-1))
        pyramid_imgs = [test_img]
        for sacled_img in pyramid_imgs:
            if np.min(sacled_img.shape[0:2])<np.min(window_size):
                break
            for (x,y,window_img) in self.sliding_window(sacled_img,window_size,step_size):
                if tuple(window_img.shape[0:2]) != window_size:
                    continue
                hist_img =  hog(window_img, orientations=9, pixels_per_cell=(6, 6),\
                                cells_per_block=(2, 2),block_norm='L1', visualize=False,\
                                transform_sqrt=False,feature_vector=True,channel_axis=-1)
                if self.model.predict(hist_img.reshape(1,-1)) == 0:#if the prediction result is phone
                    accuracy = self.model.predict_proba(hist_img.reshape(1,-1))[0][0]
                    if accuracy >= TRUST_THRESHOLD:
                        box = (x,y,x+window_size[0],y+window_size[1])
                        possible_boxes[box] = accuracy
        #pop the largest possibility bounding box
        box = possible_boxes.top()
        x1,y1,x2,y2 = box
        phone_position= np.float32(((x2+x1)/2/width,(y1+y2)/2/height))
        phone_position = np.around(phone_position,4)
      
        if visualize:
            cv2.putText(img,str(phone_position),(x1,y1-15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.imshow('bounding_box',img) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return phone_position

    def find_phone_whole_folder(self,visualize=False):
        '''
        Detect all the images under the test_folder, and print the phone position of each detection

        Input:  1.visualize: True to enable the box visualization of phone detection, default False

        Output: 1.label_dict: dictionary of ground truth phone position
                2.predict_dict: dictionary of predicted phone position
        '''
        label_file_name = os.path.join('find_phone', 'labels.txt')
        label_file = open(label_file_name)
        label_lines = label_file.readlines()
        label_dict = {}
        for line in label_lines:
            line = line.strip('\n')
            temp = line.split(" ")
            label_dict[temp[0]] = np.array([float(temp[1]),float(temp[2])])

        predict_dict = {}
        for filename in os.listdir(self.folder):
            t0 = tic()
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(self.folder,filename))

                width = int(img.shape[1] * SCALE_PERCENTAGE / 100)
                height = int(img.shape[0] * SCALE_PERCENTAGE / 100)
                dim = (width, height)
                # resize image
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                # segment the image
                phone_pos = self.find_phone_box(resized,visualize=visualize)
                print('Image '+str(filename)+' phone at: '+ str(phone_pos))
                toc(t0,'detection')
                predict_dict[filename] = np.array([phone_pos[0],phone_pos[1]])

        return label_dict,predict_dict
    
    def find_phone_single_img(self,img_path,visualize):
        img = cv2.imread(img_path)
        phone_pos = self.find_phone_box(img,visualize=visualize)
        print(phone_pos[0], " ", phone_pos[1])
    
    def cal_accuracy(self,label_dict,predict_dict):
        '''
        Given gound truth and predicted result, return the model accuracy of all
        '''
        error = []
        for result in predict_dict:
            predict_pos = predict_dict[result]
            truth_pos   = label_dict[result]
            x_error = np.abs(truth_pos[0]-predict_pos[0])/truth_pos[0]
            y_error = np.abs(truth_pos[1]-predict_pos[1])/truth_pos[1]
            error.append((x_error+y_error)/2)
        total_error = np.float32(error)
        return 1- np.round(total_error.mean(),4)

if __name__ == "__main__":
    test_folder = 'find_phone_test_images'
    model_name = 'trained_model.sav'
    loaded_model = joblib.load(model_name)
    my_detector  = PhoneDetector(folder=test_folder,model=loaded_model)

    if ACCURACY:
        label_dict,predict_dict = my_detector.find_phone_whole_folder(visualize=VISUALIZE)
        accuracy = my_detector.cal_accuracy(label_dict,predict_dict)
        print('model accuracy is: '+ str(accuracy))

    if SINGLE_TEST:
        image_path = sys.argv[1]
        current_path = os.getcwd()
        my_detector.find_phone_single_img(os.path.join(current_path, image_path,visualize=VISUALIZE))
    
    
    
