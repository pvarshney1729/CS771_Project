import numpy as np
import cv2
import os
from PIL import Image
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
# import keras 
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow as tf

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.

# size of cropped image which is entered into CNN
size = 70

def findLetter(index):
	output_data = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']	
	return output_data[index]

def decaptcha( filenames ):
	# numChars = 3 * np.ones( (len( filenames ),) )
	# # The use of a model file is just for sake of illustration
	# file = open( "model.txt", "r" )
	# codes = file.read().splitlines()
	# file.close()
	numChars = np.ones( len(filenames) )
	codes = []

	output_data = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	represent = {}
	em = []
	for i in output_data:
		em.append(0)
	for i in range(26):
		x = []
		for j in em:
			x.append(j)
		x[i] = 1
		represent[output_data[i]] = x
	# represent["A"] = one hot vector for A
			

	# we load the model here
	jsonFile = open("model.json", "r")
	loaded_model_json = jsonFile.read()
	jsonFile.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("weights.h5")
	loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	a = np.ones((150,600))*255
	b = np.ones((150,600))*127
	c = np.ones((150,600))*76
	d = np.ones((150,600))*204
	m = HSV2 = np.stack([np.zeros(150)]*3,axis = 1)
	new_img = np.stack([np.ones([140,140]),np.ones([140,140]),np.ones([140,140])],axis = 2)
	new_img = new_img*255	
	alpha = [0]*26
	count = 0
	max_count = 2000
	numFile = 0
	for file in filenames:
		captchaString = ""		
		img = cv2.imread(file)
		HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		a_match = (HSV_img[:,:,2] == a)
		b_match = (HSV_img[:,:,1] == b)
		z = (np.invert(a_match*b_match))*255		
		HSV2 = np.stack([z,z,z], axis = 2)
		HSV2 = np.asarray(HSV2)
		k = a_match*np.logical_or(b_match,(HSV_img[:,:,1] == c))
		z = (np.logical_or(k,(HSV_img[:,:,2] == d)))*255
		HSV1 = np.stack([z,z,z], axis = 2)
		HSV1 = np.asarray(HSV1)
		sum_list = np.sum(abs(HSV2[:,:,1]/255 - 1), axis=0)
		count_of_alpha = 0
		left_endpoints = []
		right_endpoints = []
		for i in range(4,597):
			if( (sum_list[i-1] == 0) and (sum_list[i-2] == 0) and (sum_list[i-3] == 0)   and (not( sum_list[i-4] == 0)) and  ((sum_list[i+1] == 0))  and (( sum_list[i+2] == 0)) and  (( sum_list[i+3] == 0))  ):
				count_of_alpha += 1
				inc = i+9
				if(i>=591):
					inc = 599
				right_endpoints.append(inc)
				
		count += 1
				
		for i in range(3,596):
			if( (sum_list[i+1] == 0) and (sum_list[i+2] == 0) and (sum_list[i+3] == 0)   and (not( sum_list[i+4] == 0)) and  ((sum_list[i-1] == 0))  and (( sum_list[i-2] == 0)) and  (( sum_list[i-3] == 0))  ):
				inc = i-7
				if(i<=7):
					inc = 0
				left_endpoints.append(inc)
		
		numChars[numFile] = count_of_alpha
		numFile = numFile + 1

		for i in range(count_of_alpha):
			width = right_endpoints[i]-left_endpoints[i]
			HSV_new = np.copy(new_img)
			HSV_new[:,(69-int(width/2)):(69-int(width/2)+int(width)+1),:] = HSV1[5:145,left_endpoints[i]:right_endpoints[i]+1,:]
			HSV_new = HSV_new / 255
			HSV_new = cv2.resize(HSV_new, (size,size)) 
			h, s, HSV_new =  cv2.split(HSV_new)
			HSV_new = HSV_new[..., np.newaxis]
			temp = []
			temp.append(HSV_new)
			temp = np.array(temp)
			captchaString = captchaString + findLetter(loaded_model.predict_classes(temp)[0])
		codes.append(captchaString)
		# print(captchaString)
	return (numChars, codes)



