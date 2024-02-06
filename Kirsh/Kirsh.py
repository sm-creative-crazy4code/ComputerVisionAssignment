import cv2
import numpy as np


KIRSCH_K1   = np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float32) / 15
KIRSCH_K2   = np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float32) / 15
KIRSCH_K3   = np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float32) / 15
KIRSCH_K4   = np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float32) / 15
KIRSCH_K5   = np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float32) / 15
KIRSCH_K6   = np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float32) / 15
KIRSCH_K7   = np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float32) / 15
KIRSCH_K8   = np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float32) / 15


def kirsch_filter(img) :
    """Expects a grayscale image and returns a gray-scale image that's been Kirsch edge filtered. """
    
    fimg    = np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K1),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K2),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K3),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K4),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K5),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K6),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K7),
                            cv2.filter2D(img, cv2.CV_8U, KIRSCH_K8),
                           )))))))
    return(fimg)


def cartoonize(img_name):

    # load the image
    image_to_animate=cv2.imread(img_name)

    # smoothening the image while preserving the edges
    smoothened_image=cv2.GaussianBlur(image_to_animate,(3,3),0)
    
    # grayscale conversion
    gray_image=cv2.cvtColor(smoothened_image,cv2.COLOR_BGR2GRAY)

    
    
    edge_mask=kirsch_filter(gray_image)
    cv2.imshow("Kirsh_mask",edge_mask)
    cv2.imwrite("01_Kirsh_mask.jpeg",edge_mask)
       


    # inverted mask
    inverted_edge_mask = cv2.bitwise_not(edge_mask)
    cv2.imshow("Inverted_mask",inverted_edge_mask)
    cv2.imwrite('02_Inverted_Mask.jpeg',inverted_edge_mask) 

    

    
    # finding adapting thereshold values for the edge mask
    edges=cv2.adaptiveThreshold(inverted_edge_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    # combining smoothened image and edge maskS
    cartoon_image=cv2.bitwise_and(smoothened_image,smoothened_image,mask=edges )

    
    cv2.imshow("Cartooned Image", cartoon_image)
    cv2.imwrite('03_Cartooned_Image.jpeg',cartoon_image)  

    cv2.waitKey(0)
    cv2.destroyAllWindows()


cartoonize("cakeimg.png")






