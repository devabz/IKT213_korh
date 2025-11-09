import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def save(path, img):
    return cv2.imwrite(path, img)

def padding(image, border_width): 
    # http://www.bim-times.com/opencv/3.3.0/d3/df2/tutorial_py_basic_ops.html
    return cv2.copyMakeBorder(
        src=image, 
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT
    )

def crop(image, x_0, x_1,  y_0, y_1): 
    # https://learnopencv.com/cropping-an-image-using-opencv/
    return image[y_0:-y_1, x_0:-x_1]

def resize(image, width, height): 
    # https://www.geeksforgeeks.org/python/image-resizing-using-opencv-python/
    return cv2.resize(image, (height, width))

def copy(image, emptyPictureArray): 
    # flag = not np.may_share_memory(emptyPictureArray, image) # Should be False
    emptyPictureArray[:] = image[:]
    return emptyPictureArray

def grayscale(image): 
    # https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def hsv(image): 
    # https://techtutorialsx.com/2019/11/08/python-opencv-converting-image-to-hsv/
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def hue_shifted(image, emptyPictureArray, hue): 
    # emptyPictureArray[:] = np.clip(image[:].astype(int) + hue, a_min=0, a_max=255).astype(np.uint8)
    # return emptyPictureArray

    # ChatGPT
    img_hsv = hsv(image).astype(np.uint16)
    
    # Shift hue channel (hue values wrap around at 180 in OpenCV)
    img_hsv[..., 0] = (img_hsv[..., 0] + hue) % 180
    
    # Convert back to uint8 and RGB
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    shifted = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    # Copy into pre-allocated array
    emptyPictureArray[:] = shifted
    return emptyPictureArray

def smoothing(image): 
    # https://analyticsindiamag.com/complete-tutorial-on-linear-and-non-linear-filters-using-opencv/%C2%A0
    return cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    
def rotation(image, rotation_angle): 
    # https://www.geeksforgeeks.org/python/python-opencv-cv2-rotate-method/
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)


if __name__ == '__main__':
    img = cv2.imread('lena-2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = dict(
        original=img,
        padded=padding(img, border_width=100),
        cropped=crop(
            img, 
            x_0=80,  # From the left
            x_1=130, # From the right
            y_0=80,  # From the top
            y_1=130, # From the bottom
        ),
        resized=resize(img, 200, 200),
        copied=copy(img, np.empty_like(img)),
        smooth=smoothing(img),
        rotation_90=rotation(img, 90),
        rotation_180=rotation(img, 180),
        grayscaled=grayscale(img),
        hsv=hsv(img),
        hue_shifted=hue_shifted(img, np.empty_like(img), hue=50),
    )


    path = 'imgs/'
    os.makedirs(path, exist_ok=True)
    for k, v in img.items():
        k = os.path.join(path, f'{k}.png')
        v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
        save(path=k, img=v)

