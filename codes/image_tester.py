from PIL import Image
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import resize


img = Image.open('a.jpg')
img = np.array(img)
print(img.shape)
print(np.max(img))
print(np.min(img))