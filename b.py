import json
from pprint import pprint
import cv2

annos = json.load(open('train_annotations.json'))

pprint(annos['images'][1])


img = 'data/FB_IMG_1628955610417.jpg'
img = cv2.imread(img)

cv2.imshow(img)
cv2.waitKey(0)