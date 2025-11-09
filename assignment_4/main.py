import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save(path, img):
    return cv2.imwrite(path, img)

def harris_corner_detection(reference_image):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img = reference_image.copy()
    
    msk = dst > 0.01 * dst.max()

    img[msk] = [0, 0, 255]

    return {'harris': img}


def align_image(image_to_align, reference_image, max_features, good_match_percent):
    # https://github.com/spmallick/learnopencv/blob/master/ImageAlignment-FeatureBased/align.py 
    
    align_img = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    MIN_MATCH_COUNT = 10

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(align_img, None)
    kp2, des2 = sift.detectAndCompute(reference, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(
        n_features=max_features,
        algorithm=FLANN_INDEX_KDTREE,
        trees=5
    )

    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)


    good = []
    for m, n in matches:
        if m.distance < good_match_percent*n.distance:
            good.append(m)
    

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = align_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        img2 = cv2.polylines(reference,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
 
    img3 = cv2.drawMatches(align_img,kp1,img2,kp2,good,None,**draw_params)
    
    dim = (len(good), 2)

    p1 = np.zeros(dim, dtype=np.float32)
    p2 = np.zeros(dim, dtype=np.float32)

    for i, match in enumerate(good):
        p1[i, :] = kp1[match.queryIdx].pt
        p2[i, :] = kp2[match.trainIdx].pt
    
    h, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    height, width, c = reference_image.shape

    img1Reg = cv2.warpPerspective(image_to_align, h, (width, height))
    return {'aligned': img1Reg, 'matches': img3}


if __name__ == '__main__':
    align_img = cv2.imread('align_this.jpg')
    reference = cv2.imread('reference_img.png')

    matches = harris_corner_detection(reference)
    aligned = align_image(
        align_img, reference, 10, 0.7
    )

    imgs = {**matches, **aligned, 'align_this': align_img, 'reference': reference}

    path = 'imgs/'
    os.makedirs(path, exist_ok=True)
    for k, v in imgs.items():
        k = os.path.join(path, f'{k}.png')
        save(path=k, img=v)

