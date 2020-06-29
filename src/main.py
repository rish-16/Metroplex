import cv2, math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from shape import Shape

def read_image(path):
    img = Image.open(path)
    
    constraints = {}
    constraints['width'] = max(np.array(img).shape[:2])
    constraints['height'] = min(np.array(img).shape[:2])
    
    return img, constraints
    
# standard MSE loss
def get_loss(img1, img2):
    diff = np.square(img1 - img2)
    cost = np.sum(diff)
    cost = float(cost / (img1.shape[0] * img1.shape[1] * img1.shape[2]))
    
    return cost
    
def draw_angled_rec(x0, y0, x1, y1, colour, angle, img):
    _angle = angle * math.pi / 180.0
    width = x1 - x0
    height = y1 - y0
    
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, colour, -1)
    cv2.line(img, pt1, pt2, colour, -1)
    cv2.line(img, pt2, pt3, colour, -1)
    cv2.line(img, pt3, pt0, colour, -1)
    
    return img
    
img, constraints = read_image("../assets/sky.jpg")
by_color = defaultdict(int)
for pixel in img.getdata():
    by_color[pixel] += 1
constraints['all_colours'] = list(by_color.keys())

img = np.array(img)
average_colour = img.mean(axis=0).mean(axis=0)

n_shapes = 100

losses = []

canvas = np.full(shape=img.shape, fill_value=average_colour, dtype=np.int8)
for i in range(n_shapes):
    cur_shape = Shape(constraints)
    
    canvas = cur_shape.superimpose(canvas)
    
    loss = get_loss(img, canvas)
    print (loss)
    losses.append(loss)
    
plt.figure(1)
plt.subplot(121)
plt.imshow(canvas)
plt.subplot(122)
plt.imshow(img)
plt.show()

plt.plot(range(n_shapes), losses, color="green")
plt.show()