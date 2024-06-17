import cv2
from PIL import Image
import numpy as np

def main():
    
    img = cv2.imread('images/2.jpeg')
    resized = cv2.resize(img, (1200,1500), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/2-1.pnm', resized)
    
    resized = cv2.resize(img, (1800,2400), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/2-2.pnm', resized)
    
    resized = cv2.resize(img, (3000,3000), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/2-3.pnm', resized)
    
    resized = cv2.resize(img, (3600,5400), interpolation= cv2.INTER_LINEAR)
    
    cv2.imwrite('images/2-4.pnm', resized)


    img = Image.open('/home/rubs/Documents/code/HPC/images/1.eps').convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1].copy()
    resized = cv2.resize(img, (1200,1500), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/1-1.pnm', resized)
    
    resized = cv2.resize(img, (1800,2400), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/1-2.pnm', resized)
    
    resized = cv2.resize(img, (3000,3000), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/1-3.pnm', resized)
    
    resized = cv2.resize(img, (3600,5400), interpolation= cv2.INTER_LINEAR)
    
    cv2.imwrite('images/1-4.pnm', resized)

    img = cv2.imread('images/3.png')
    resized = cv2.resize(img, (1200,1500), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/3-1.pnm', resized)
    
    resized = cv2.resize(img, (1800,2400), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/3-2.pnm', resized)
    
    resized = cv2.resize(img, (3000,3000), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite('images/3-3.pnm', resized)
    
    resized = cv2.resize(img, (3600,5400), interpolation= cv2.INTER_LINEAR)
    
    cv2.imwrite('images/3-4.pnm', resized)

if __name__ == "__main__":
    main()
