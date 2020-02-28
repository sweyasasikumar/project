import cv2
import numpy as np

def getTextOverlay(input_image):
    data = np.zeros(input_image.shape, dtype=np.uint8)

    #convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #range of blackcolour
    lower_val = np.array([0,0,0])
    upper_val = np.array([179,100,130])

    #threshold the hsv image to get black colour
    mask = cv2.inRange(hsv, lower_val, upper_val)

    # invert mask to get black letters on white background
    output = cv2.bitwise_not(mask)
    return output

if __name__ == '__main__':
    image = cv2.imread('simpsons_frame0.png')
    output = getTextOverlay(image)
    cv2.imwrite('simpons_text.png', output)

    #display
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
