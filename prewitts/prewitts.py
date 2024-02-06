import cv2, argparse, sys
import numpy as np

def cartoonize(img_name):

    # load the image
    image_to_animate=cv2.imread(img_name)

    # smoothening the image while preserving the edges
    smoothened_image=cv2.bilateralFilter(image_to_animate,d=9,sigmaColor=75,sigmaSpace=75)
    
    # grayscale conversion
    gray_image=cv2.cvtColor(smoothened_image,cv2.COLOR_BGR2GRAY)

    
    
    #  prewitts operator(x-axis)
    kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    #  prewitts operator(y-axis)
    kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    
    # edge dectection
    # horizontal dectection
    img_prewittx = cv2.filter2D(gray_image, -1, kernelx)
    # vertical dectection
    img_prewitty = cv2.filter2D(gray_image, -1, kernely)
    # combined dectection
    edge_mask=cv2.bitwise_or(img_prewittx, img_prewitty)
    cv2.imshow("combined_mask",edge_mask)
    cv2.imwrite('01_Combined_Mask.jpeg',edge_mask) 


       


    # inverted mask
    inverted_edge_mask = cv2.bitwise_not(edge_mask)
    cv2.imshow("Inverted_mask",inverted_edge_mask)
    cv2.imwrite('02_Inverted_Mask.jpeg',inverted_edge_mask) 

    

    
    # finding adapting thereshold values for the edge mask
    edges=cv2.adaptiveThreshold(inverted_edge_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    # combining smoothened image and edge maskS
    cartoon_image=cv2.bitwise_and(smoothened_image,smoothened_image,mask=edges )

    # cv2.imwrite("cartooned_x",cartoon_image)
    cv2.imshow("Cartooned Image", cartoon_image)
    cv2.imwrite('03_Cartooned_Image.jpeg',cartoon_image)  

    cv2.waitKey(0)
    cv2.destroyAllWindows()


cartoonize("cakeimg.png")



