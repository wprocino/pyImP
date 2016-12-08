import time

import numpy as np
import urllib
import cv2

print  ("OpenCV version:", cv2.__version__)
print  ("Numpy version:", np.__version__, "\n")

from PIL import Image
##from PIL import Image



# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url) -> object:
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    resp = urllib.urlopen(url)

    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image



# img = cv2.imread('c:/images/')
file = 'C:\\images\\pencilsInColor.jpg'
#img = cv2.imread(file, 0)   # reads black and white
#img = cv2.imread(file, 3)   # reads RGB color
img = cv2.imread(file, 4)   # reads color with extra channel

print("opening ", file)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 400, 300)




hist = cv2.calcHist( img, [0], None, [256],[0,256] )

#plt.hist(img.ravel(),256,[0,256]); plt.show()
#hist, bins = np.histogram(img.ravel(),256,[0,256])

##color = ('b','g','r')
##for i,col in enumerate(color):
##    histr = cv2.calcHist([img],[i],None,[256],[0,256])
##    plt.plot(histr,color = col)
##    plt.xlim([0,256])
##plt.show()



cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Setting up for video capture")
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1024)
#time.sleep(2)
#cap.set(15, -8.0)

ret = 0
counter = 0
bGrabbing = 1
while (bGrabbing == 1 and counter < 10):
    # ret, img = cap.read()
    # print ("capture returns: ", ret, " ", counter)

    img = cap.read()

    ## if(img > 0): cv2.imshow("input", img)
    #cv2.imshow("thresholded", imgray*thresh2)

    key = cv2.waitKey(10)
    if key == 27:
        bGrabbing = 0
        break

    counter+=1
    time.sleep(0.20)

cv2.VideoCapture(0).release()

#trying for a web image
print("trying for a web image")
url = " https://localrooted.files.wordpress.com/2016/09/20160915_135051.jpg"

image = url_to_image(url)
#resp = urllib.request(url)
# img = np.array(Image.open(StringIO(response.content)))

cv2.imshow('web image',image)
cv2.destroyAllWindows()

print("OK, done.")

