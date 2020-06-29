import cv2, math, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
from shapes import Shape

def read_image(path):
    img = Image.open(path)
    
    constraints = {}
    constraints['width'] = max(np.array(img).shape[:2])
    constraints['height'] = min(np.array(img).shape[:2])
    
    return img, constraints
    
# def mutate(shape):
#     theta = 0.4
#     if np.random.uniform(0, 1) < theta:
#         shape.x = np.random.randint(shape.constraints['width'])
#     if np.random.uniform(0, 1) < theta:
#         shape.y = np.random.randint(shape.constraints['height'])
#     if np.random.uniform(0, 1) < theta:
#         shape.rad = np.random.randint(shape.constraints['width'])
#     if np.random.uniform(0, 1) < theta:
#         shape.alpha = float('{:.2f}'.format(np.random.uniform(0, 1)))
#     if np.random.uniform(0, 1) < theta:
#         shape.colour = random.choice(shape.constraints['all_colours'])
    
#     return shape
    
target, constraints = read_image("../assets/sky.jpg")
constraints['all_colours'] = list(target.getdata())

target = np.array(target, dtype=np.int8)
average_colour = target.mean(axis=0).mean(axis=0)
# canvas = np.full(shape=target.shape, fill_value=average_colour, dtype=np.int8)
canvas = np.zeros_like(target, dtype=np.uint8)
canvas[:] = 255
canvas = canvas.astype(np.int8)

# hyper params
T_max = 250
T_min = 1
delta_T = 0.9
temps = np.arange(T_max, T_min, -delta_T)
theta = 0.2
all_losses = []
all_images = []

# simulated annealing
for i in range(len(temps)):
    T = temps[i]
    
    shape = Shape(constraints) # random shape
    
    canvas_with_shape, canvas = shape.superimpose(canvas) # C -> N
    score = shape.get_score(target, canvas_with_shape) # ∆E
    
    all_losses.append(score)
    
    if score > theta: # accept good shapes only
        canvas = canvas_with_shape
    elif np.exp(-score / T) > np.random.uniform(0, 1): # accept any reasonable shape
        canvas = canvas_with_shape
    else:
        pass # shape has failed -> will not be part of generated image
    
    all_images.append(canvas)
    
# animating image generation process
fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

for i in range(len(all_images)):
    all_images[i] = [plt.imshow(all_images[i], animated=True)]
    
anim = animation.ArtistAnimation(fig, all_images, blit=True, repeat_delay=5000)
plt.show()

# plt.plot(temps, all_losses, color="green")
# plt.show()