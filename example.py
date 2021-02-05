from recognition import E2E
import cv2
from pathlib import Path
import argparse
import time
import os

def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-v', '--video_name',
                     help='link to image', default='')

    return arg.parse_args()


input_folder = './videos'
output_folder = './results'
args = get_arguments()
video_name = args.video_name
input_path = os.path.join(input_folder, video_name)
output_path = os.path.join(output_folder, video_name)
# read image
# start

# load model
model = E2E()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
i = 0
cap = cv2.VideoCapture(input_path)
ret, frame = cap.read()
height, width, channels = frame.shape
video = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

while cap.isOpened():
    try:
        ret, frame = cap.read()
        # recognize license plate
        image = model.predict(frame)
        video.write(image)
        # show image
        cv2.imshow('License Plate', image)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        i += 1
        print(i)
        if i == 200:
            break
    except Exception as e:
        print(e)
        continue

cv2.destroyAllWindows()
video.release()
