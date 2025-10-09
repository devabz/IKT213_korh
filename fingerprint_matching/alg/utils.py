import os
import cv2
import json
import time
import psutil
import tracemalloc
from itertools import chain
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def run_and_trace(f):
    process = psutil.Process(os.getpid())
    c_before = process.cpu_times()
    m_before = process.memory_info().rss
    tracemalloc.start()

    t_before = time.time()
    out = f()
    t_after = time.time()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    c_after = process.cpu_times()
    m_after = process.memory_info().rss

    scale = (1024 ** 2)
    m_diff = m_after - m_before  
    peak_memory = peak / scale

    logs = dict(
        cpu_user=c_after.user - c_before.user,
        cpu_sys=c_after.system - c_before.system,
        wall_time=t_after-t_before,
        mem_before=m_before / scale,
        mem_after=m_after / scale,
        mem_diff=m_diff / scale,
        peak_mem_MB=peak_memory
    )
    return out, logs

def listdir(path):
    if not os.path.isdir(path): return None
    contents = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
    return contents


def timestamp():
    return datetime.now().strftime('%y%m%d%H%M%S')

def write_json(path, content):
    with open(path, 'w') as f:
        json.dump(content, f, indent=4)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def preprocess_fingerprint(image_path, gray_scale=True):
    if not gray_scale:
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin

def process_dataset(dataset_path, results_folder, match_fn, name=None, save_imgs=True, plot_img=False, verbose=False):
    threshold = 20  # Adjust this based on tests
    y_true = []  # True labels (1 for same, 0 for different)
    y_pred = []  # Predicted labels

    os.makedirs(os.path.join(results_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(results_folder, 'stats'), exist_ok=True)
    if save_imgs:
        os.makedirs(os.path.join(results_folder, 'keypoints'), exist_ok=True)
    
    
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):  # Check if it's a valid directory
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.png', '.jpg'))]
            if len(image_files) != 2:
                print(f"Skipping {folder}, expected 2 images but found {len(image_files)}")
                continue
            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])
            (match_count, match_img), logs = match_fn(img1_path, img2_path)
            logs['path'] = folder_path
 
            # Determine the ground truth
            actual_match = 1 if "same" in folder.lower() else 0  # 1 for same, 0 for different
            y_true.append(actual_match)
 
            # Decision based on good matches count
            predicted_match = 1 if match_count > threshold else 0
            y_pred.append(predicted_match)
            result = "matched" if predicted_match == 1 else "unmatched"

            stats = dict(folder=folder, result=result, match_count=match_count)
            
            if verbose:
                print(f"{folder}: {result.upper()} ({match_count} good matches)")
            
            write_json(os.path.join(results_folder, 'logs', f'{folder}_{result}.json'), logs)
            write_json(os.path.join(results_folder, 'stats', f'{folder}_{result}.json'), stats)

            if match_img is not None and save_imgs:
                match_img_filename = f"{folder}_{result}.png"
                match_img_path = os.path.join(results_folder, 'keypoints', match_img_filename)
                cv2.imwrite(match_img_path, match_img)
                print(f"Saved match image at: {match_img_path}")
            
 
    # Compute and display confusion matrix
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title(f"Confusion Matrix {name or ''}".strip())
    plt.savefig(os.path.join(results_folder, 'cfm.png'))
    if not plot_img: plt.close(fig)
    write_json(os.path.join(results_folder, 'cm.json'), cm.tolist())



