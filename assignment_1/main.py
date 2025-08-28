import cv2
import time
import numpy as np


def print_image_information(image: np.ndarray):
    height, width, channels = image.shape

    info = {
        'height':height,
        'width':width,
        'channels':channels,
        'size':image.size,
        'data type': image.dtype
    }

    print('\n'.join(f"{k}: {v}" for k, v in info.items()))


def compute_fps(cam, num_frames):
    start = time.time() 
    for _ in range(num_frames):
        cam.read()
    end = time.time()
    return  num_frames / (end - start)

def write_web_camera_information_to_txt():
    cam = cv2.VideoCapture(0)
    fps = compute_fps(cam, num_frames=60)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam.release()
    cv2.destroyAllWindows()

    info = {
        'fps': round(fps, 2),
        'height': frame_height,
        'width': frame_width,
    }

    txt = '\n'.join(f"{k}: {v}" for k, v in info.items())

    with open('assignment_1/solutions/camera_outputs.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    img_path = 'assignment_1/lena-1.png'
    image = cv2.imread(img_path)
    print_image_information(image)
    write_web_camera_information_to_txt()



