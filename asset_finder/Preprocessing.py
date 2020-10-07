import numpy as np
import cv2
from PIL import Image
from pytesseract import Output
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class Preprocessing:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.slice_matrix = None 
        
    def get_array(self):
        return np.array(self.image)
    
    
        
    def convert_to_binary(self,change=False):
        img_array = self.get_array()
        rest, thresh2 = cv2.threshold(img_array,10,230,cv2.THRESH_BINARY)
        if not change:
            return thresh2
        else:
            self.image = thresh2
    
    def horizontal_slice_matrix(self):
        
        image_array = self.image
        d_i = 0
        d_j = 1
        print(np.sum(image_array))
        count = 0
        inslice = False
        n = image_array.shape[1]
        m = image_array.shape[0]
        B = np.zeros((m,n))
        i,j=0,0
        while(  i< m   ):
            if image_array[i][j] != 0:
                #print("Yes")
                count += 1
                inslice = True
            elif inslice:
                inslice = False
                i_s = i
                j_s = j
                steps = count 
                while(steps!=0):
                    steps = steps -1 
                    i_s = i_s - d_i
                    j_s = j_s - d_j
                    B[i_s][j_s] = count
                B[i][j] = count
                count = 0
            i = i + d_i
            if (j + d_j >= n):
                j = 0
                i += 1
            else:
                
                j = j + d_j
            
        return B
            
    def vertical_slice_matrix(self):
        d_i = 1
        d_j = 0
        image_array = self.image
        
        count = 0
        inslice = False
        n = image_array.shape[1]
        m = image_array.shape[0]
        B = np.zeros((m,n))
        i,j=0,0
        while(  j< n   ):
            if image_array[i][j] != 0:
                #print("Yes")
                count += 1
                inslice = True
            elif inslice:
                inslice = False
                i_s = i
                j_s = j
                steps = count 
                while(steps!=0):
                    steps = steps -1 
                    i_s = i_s - d_i
                    j_s = j_s - d_j
                    B[i_s][j_s] = count
                B[i][j] = count
                count = 0
            i = i + d_i
            if (i + d_i >= m):
                i = 0
                j += 1
            else:
                
                j = j + d_j
            
        return B
    
    def get_thickness_matrix(self):
        if self.slice_matrix == None:
            self.slice_matrix = self.get_slice_matrix()
        m = self.image.shape[0]
        n = self.image.shape[1]
        M = np.zeroes((m,n))
        for i in range(m):
            for j in range(n):
                m = self.slice_matrix[i,j,:]
                if  m != [0,0,0]:
                    ph,pv = m[0],m[1]
                    if ((ph == pv) or ((ph > 1) and 2*ph < pv)):
                        M[i,j] = ph
                    else:
                        M[i,j] = (ph + pv) / 2.0
        return M
    
    def get_slice_matrix(self):
        self.threshold(change=True)
        
        m = self.image.shape[0]
        n = self.image.shape[1]
        ph = self.horizontal_slice_matrix()
        pv = self.vertical_slice_matrix()
       
        self.slice_matrix = np.dstack((ph,pv))
        return self.slice_matrix
    
    def binary_erosion(self, nerosions=4, ndilations=3, kernel_size=2): #This step helps in extracting all the thick wall components. 
        image = self.convert_to_binary()
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        erosion = cv2.erode(image,kernel, iterations = nerosions)
        dilation = cv2.dilate(erosion, kernel, iterations = ndilations )
        
        return dilation
    def get_contours(self):
        text = self.image

        _, contours, _ = cv2.findContours(text, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        ctr = np.zeros(text.shape, dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                cv2.drawContours(ctr, contour, -1, (255, 0, 0), 2)
        return ctr




class TextReading(Preprocessing):
    def detect_text_boxes(self, boxes=False):
        img = self.image
        h, w = img.shape # assumes color image

       
        boxes = pytesseract.image_to_boxes(img, lang='eng', \
                config='--psm 11') # also include any config options you use

        
        for b in boxes.splitlines():
            b = b.split(' ')
            img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 0, 0), 2)

       
        if not boxes:
            return img
        else:
            return boxes
    
    def remove_text_boxes(self):
        img = self.image
        h, w = img.shape
        img = Image.fromarray(img)
        boxes = self.detect_text_boxes(boxes=True)
        
        
        for b in boxes.splitlines():
            b = b.split(' ')
            blank_image = 255*np.ones((int(b[3])-int(b[1]), int(b[4])-int(b[2])))
            blank = Image.fromarray(blank_image)
            img.paste(blank, (int(b[1]), h-int(b[4])))
        self.image = img
        return img
            
        
