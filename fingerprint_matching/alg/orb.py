import os
import cv2
from functools import partial
from alg.utils import (
    preprocess_fingerprint, 
    process_dataset,
    write_json,
    timestamp,
    run_and_trace
)


def match_fingerprints(img1_path, img2_path, n_features=1000, thr=0.7, k=2, gray_scale=True):
    img1, t1_pre = run_and_trace(lambda: preprocess_fingerprint(img1_path, gray_scale=gray_scale))
    img2, t2_pre = run_and_trace(lambda: preprocess_fingerprint(img2_path, gray_scale=gray_scale))
 
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)
 
    # Find keypoints and descriptors
    (kp1, des1), t1_detect = run_and_trace(lambda: orb.detectAndCompute(img1, None))
    (kp2, des2), t2_detect = run_and_trace(lambda: orb.detectAndCompute(img2, None))
    
    if des1 is None or des2 is None:
        return 0, None  # Return 0 matches if no descriptors found
 
    # Use Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
 

 
    # KNN Match
    matches, t_match = run_and_trace(lambda: bf.knnMatch(des1, des2, k=k))
 
    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < thr * n.distance]
 
    # Draw only good matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    results =  len(good_matches), match_img
    
    logs = dict(
        preprocessing=[t1_pre, t2_pre],
        detection=[t1_detect, t2_detect],
        matching=t_match
    )
    
    return results, logs




if __name__ == '__main__':
    dataset_path = 'path/to/load/data'
    results_folder = 'path/to/store/results'
        
    dataset_path   = 'data_check-20250922T083054Z-1-001/data_check'
    results_folder = 'results/orb'
    results_folder = os.path.join(results_folder, timestamp())

    assert os.path.isdir(dataset_path), f"Path does not exist {dataset_path}"

    match_params = dict(
        n_features=1000,
        thr=0.70,
        k=2,
    )

    process_dataset(
        dataset_path=dataset_path, 
        results_folder=results_folder,
        match_fn=partial(match_fingerprints, **match_params),
        name='ORB+BF'
    )

    write_json(os.path.join(results_folder, 'match_params.json'), match_params)