
import cv2
import numpy as np

def resize_image(img, percent):
    w=int(img.shape[1]*percent/100); h=int(img.shape[0]*percent/100)
    return cv2.resize(img,(w,h))

def adjust_brightness(img, value):
    return cv2.convertScaleAbs(img, alpha=1, beta=value)

def adjust_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def grayscale(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g,cv2.COLOR_GRAY2RGB)

def blur(img):
    return cv2.GaussianBlur(img,(11,11),0)

def warm_filter(img):
    b,g,r=cv2.split(img)
    r=cv2.add(r,20); b=cv2.subtract(b,10)
    return cv2.merge((b,g,r))

def portrait_blur(img):
    blur_img=cv2.GaussianBlur(img,(31,31),0)
    h,w=img.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    cv2.ellipse(mask,(w//2,h//2),(w//4,h//3),0,0,360,255,-1)
    mask=cv2.GaussianBlur(mask,(51,51),0)
    mask=mask.astype(float)/255
    out=np.empty_like(img)
    for i in range(3):
        out[:,:,i]=(img[:,:,i]*mask + blur_img[:,:,i]*(1-mask)).astype(np.uint8)
    return out

def sharpen(img):
    k=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img,-1,k)

def edge_detection(img):
    e=cv2.Canny(img,100,200)
    return cv2.cvtColor(e,cv2.COLOR_GRAY2RGB)

def sketch(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    inv=255-gray
    blur=cv2.GaussianBlur(inv,(21,21),0)
    invb=255-blur
    s=cv2.divide(gray,invb,scale=256)
    return cv2.cvtColor(s,cv2.COLOR_GRAY2RGB)

def cartoon(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray=cv2.medianBlur(gray,5)
    edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    color=cv2.bilateralFilter(img,9,250,250)
    return cv2.bitwise_and(color,color,mask=edges)

def rotate_image(img,angle):
    h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    return cv2.warpAffine(img,M,(w,h))
