import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import random

# get file names of frames
col_frames= sorted(glob.glob(os.path.join('/home/yihong/lane_keeping/frames/', "*.png")))
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

index = random.randint(0, 100)

print(col_frames[index])
print(np.shape(col_frames))

col_images = []

for frame in col_frames:
    img = cv2.imread(frame)
    col_images.append(img)


# create a zero array
stencil = np.zeros_like(col_images[index][:,:,0])

# specify coordinates of the polygon
polygon = np.array([[50,270], [220,160], [360,160], [480,270]])

# fill polygon with ones
cv2.fillConvexPoly(stencil, polygon, 1)

# apply polygon as a mask on the frame
img = cv2.bitwise_and(col_images[index][:,:,0], col_images[index][:,:,0], mask=stencil)

# apply image thresholding
ret, thresh = cv2.threshold(img, 130, 145, cv2.THRESH_BINARY)

#hough line transformation
lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)

# create a copy of the original frame
dmy = col_images[index][:,:,0].copy()

# draw Hough lines
for line in lines:
  x1, y1, x2, y2 = line[0]
  cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)

# plot frame
plt.figure(figsize=(10,10))
plt.imshow(dmy, cmap= "gray")
plt.show()

