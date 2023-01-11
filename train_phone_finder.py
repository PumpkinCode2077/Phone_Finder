import sys
import cv2
import random
import time
import numpy as np
import os
import cv2
import sklearn.svm as SVM
import joblib
from skimage.feature import hog
'''
The python program to extract and train the data, This model extract the HOG feature of 
the image and apply an sklearn SVM model for classification.

Author: Shusen Lin
Date: 1/9/2023
'''
corp_window_size   = 22
NEG_SAMPLE_PER_IMG = 30 #change to modify the number of negative samples

def tic():
  return time.time()

def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def select_region(img, pos, num_backgrounds=20):
    '''
    Given a single phone included image, 
    
    Input:  1.img: 490x326 pixels original image
            2.pos: ground truth phone position (x,y)
            3.num_backgrounds: desired number of negative samples per image, default as 20

    Output: 1.phone_data: 4 samples of phone (#samples,#features)
            2.none_phone_data: non-phone samples(#samples,#features)
    '''
    img = np.array(img)
    height, width = img.shape[0:2]
    phone_data = []
    none_phone_data = []
    phone_pos = np.array([int(pos[0]*width), int(pos[1]*height)])
    #randomly pick non-phone regions
    for idx in range(num_backgrounds): 
        x_non = random.randint(corp_window_size, width-corp_window_size-1)
        y_non = random.randint(corp_window_size, height-corp_window_size-1)
        distance = np.linalg.norm([x_non-phone_pos[0],y_non-phone_pos[1]])
        #avoid to overlap with the phone region
        while distance < 2*corp_window_size*np.sqrt(2):
            x_non = random.randint(corp_window_size, width-corp_window_size-1)
            y_non = random.randint(corp_window_size, height-corp_window_size-1)
            distance = np.linalg.norm([x_non-phone_pos[0],y_non-phone_pos[1]])
            
        non_phone_pos = np.array([x_non,y_non])
        non_window_ub = non_phone_pos[1] - corp_window_size
        non_window_db = non_phone_pos[1] + corp_window_size
        non_window_lb = non_phone_pos[0] - corp_window_size
        non_window_rb = non_phone_pos[0] + corp_window_size
        none_phone_crop = img[non_window_ub:non_window_db,non_window_lb:non_window_rb]
        none_phone_data.append(none_phone_crop)

    window_ub  = phone_pos[1] - corp_window_size
    window_db  = phone_pos[1] + corp_window_size
    window_lb  = phone_pos[0] - corp_window_size
    window_rb  = phone_pos[0] + corp_window_size 
    phone_crop = img[window_ub:window_db,window_lb:window_rb]
    phone_data.append(phone_crop)
    phone_rotate = phone_crop
    #rotate the phone crop with 90,180,270 degrees to obtain more postive samples
    for idx in range(3):
        phone_rotate = cv2.rotate(phone_rotate, cv2.ROTATE_90_CLOCKWISE)
        phone_data.append(phone_rotate)
    
    return phone_data,none_phone_data

def preprocess_data(dir):
    '''
    The function for preprocess the data, given the image directory, collect all the ground truth 
    informations about training data, extract the HOG features, create the corresponding labels, 
    and print out the X_train and y_train
    
    Input:  1.image directory:dir - dtype:str

    Output: 1.phone train data:X_train, (HOG features) - dtype:numpy.ndarray(#samples,#features)
            2.label train data:y_train, (0:phone, 1:non-phone) - dtype:numpy.ndarray(#samples,#features)
    '''
    X_phone      = [] 
    X_none_phone = []

    y = np.array([])
    label_file_name = os.path.join(dir, 'labels.txt')
    label_file = open(label_file_name)
    label_lines = label_file.readlines()
    label_dict = {}
    for line in label_lines:
        line = line.strip('\n')
        temp = line.split(" ")
        label_dict[temp[0]] = np.array([float(temp[1]),float(temp[2])])

    for filename in os.listdir(dir):
        if filename != 'labels.txt':
            img = cv2.imread(os.path.join(dir,filename))
            phone_img, none_phone_img = select_region(img,label_dict[filename],num_backgrounds=NEG_SAMPLE_PER_IMG)
            #Extract the HOG features
            for corp in phone_img:
                hist_phone =  hog(corp, orientations=9, pixels_per_cell=(6, 6),\
                                    cells_per_block=(2, 2),block_norm='L1', visualize=False,\
                                    transform_sqrt=False,feature_vector=True,channel_axis=-1)
                X_phone.append(hist_phone)

            for corp in none_phone_img:
                hist_none_phone =  hog(corp, orientations=9, pixels_per_cell=(6, 6),\
                                    cells_per_block=(2, 2),block_norm='L1', visualize=False,\
                                    transform_sqrt=False,feature_vector=True,channel_axis=-1)
                X_none_phone.append(hist_none_phone)
    samples = X_phone + X_none_phone
    samples = np.float32(samples)

    X  = samples
    y  = np.hstack((np.zeros(len(X_phone)), np.ones(len(X_none_phone))))
    y  = y.astype(int)
    print('postive samples: ' ,len(X_phone))
    print('negative samples: ',len(X_none_phone))
    return X,y

if __name__ == "__main__":
    main_path = sys.argv[1]
    # main_path = 'find_phone'
    print('Images are processing...')
    t0 = tic()
    # Preprocess the data
    X_train,y_train= preprocess_data(main_path)
    print('Image process is done')
    toc(t0,'Image process')
    # Train
    print('Training...')
    t1 = tic()
    #use default setting of SVM classifier
    phone_clf = SVM.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, \
                  shrinking=True, probability=True, tol=0.001, cache_size=200, \
                  class_weight=None, verbose=False, max_iter=-1, \
                  decision_function_shape='ovr', break_ties=False, random_state=None)
    phone_clf.fit(X_train, y_train)
    print('Training finish')
    toc(t1,'training')
    # save the model to disk
    filename = 'trained_model.sav'
    joblib.dump(phone_clf, filename)
    print('Model saves at '+main_path+'/'+filename)

    

   
