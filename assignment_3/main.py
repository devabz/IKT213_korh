import os
import cv2
import numpy as np

def to_gray(img):

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def save(path, img):
    return cv2.imwrite(path, img)

def sobel_edge_detection(image):
    # https://learnopencv.com/edge-detection-using-opencv/

    img_gray = to_gray(image)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    return cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1)

def canny_edge_detection(image, threshold_1, threshold_2):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

    img_gray = to_gray(image)
    return cv2.Canny(img_gray, threshold_1, threshold_2)

def template_match(image, template):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    
    img_gray = to_gray(image)
    tmp_gray = to_gray(template)

    h, w = tmp_gray.shape

    res = cv2.matchTemplate(img_gray, tmp_gray, cv2.TM_CCOEFF_NORMED)

    thr = 0.9
    loc = np.where(res >= thr)
    img = image.copy()

    for pt in zip(*loc[::-1]):
        dx, dy = pt
        dx += w
        dy += h
        cv2.rectangle(img, pt, (dx, dy), (0, 0, 255), 2)

    return img


def resize(image, scale_factor: int, up_or_down: str):
    # https://docs.opencv.org/4.5.1/d4/d1f/tutorial_pyramids.html
    
    y, x, c = image.shape

    if up_or_down == "down":
        x //= scale_factor
        y //= scale_factor

        return cv2.pyrDown(image, dstsize=(x, y))
    
    elif up_or_down == "up":
        x *= scale_factor
        y *= scale_factor

        return cv2.pyrUp(image, dstsize=(x, y))


if __name__ == '__main__':
    img = cv2.imread('lambo.png')
    target = cv2.imread('shapes.png')
    template = cv2.imread('shapes_template.jpg')

    img = dict(
        original=img,
        shapes=target,
        shapes_template=template,
        sobel_edge_detection=sobel_edge_detection(img),
        canny_edge_detection=canny_edge_detection(img, 50, 50),
        template_match=template_match(target, template),
        resize=resize(img, 2, "down"),
    )


    path = 'imgs/'
    os.makedirs(path, exist_ok=True)
    for k, v in img.items():
        k = os.path.join(path, f'{k}.png')
        save(path=k, img=v)


