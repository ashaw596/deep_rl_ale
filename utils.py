try:
  from scipy.misc import imresize
except:
  import cv2
  imresize = cv2.resize

def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114])
