from pseyepy import Camera, Display
import numpy as np
import os
import json


c = Camera([0, 1, 2], fps=[80, 80, 80], resolution=Camera.RES_LARGE, colour=True)
d = Display(c) # begin the display

dirname = os.path.dirname("/home/mariobo/Autonomous-mini-drone/computer_code/api/")
filename = os.path.join(dirname, "camera-params.json")
f = open(filename)
camera_params = json.load(f)

# read from the camera/s
frame, timestamp = c.read()


def make_square(img):
    x, y, _ = img.shape
    size = max(x, y)
    new_img = np.zeros((size, size, 3), dtype=np.uint8)
    ax,ay = (size - img.shape[1])//2,(size - img.shape[0])//2
    new_img[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img

    # Pad the new_img array with edge pixel values
    # Apply feathering effect
    feather_pixels = 8
    for i in range(feather_pixels):
        alpha = (i + 1) / feather_pixels
        new_img[ay - i - 1, :] = img[0, :] * (1 - alpha)  # Top edge
        new_img[ay + img.shape[0] + i, :] = img[-1, :] * (1 - alpha)  # Bottom edge


    return new_img

for i in range(0, 1):
    frame[i] = np.rot90(frame[i], k=camera_params[0]["rotation"])
    frame[i] = make_square(frame[i])
# when finished, close the camera

c.end()