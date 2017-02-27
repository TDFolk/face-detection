import sys
from src import crop
import cv2

image = sys.argv[1]

image = crop.crop_image(image)


