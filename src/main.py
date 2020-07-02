import cv2, math, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
from shapes import Shape
from skimage.measure import compare_ssim

# standard SSIM score [-1, +1]
def get_score(og_img, gen_img):
    (score, diff) = compare_ssim(og_img, gen_img, full=True, multichannel=True)
        
    return score

def read_image(path):
    img = Image.open(path)
    
    constraints = {}
    constraints['width'] = max(np.array(img).shape[:2])
    constraints['height'] = min(np.array(img).shape[:2])
    
    return img, constraints
    
# def mutate(shape):
#     epsilon = 0.4
#     if np.random.uniform(0, 1) < epsilon:
#         shape.x = np.random.randint(shape.constraints['width'])
#     if np.random.uniform(0, 1) < epsilon:
#         shape.y = np.random.randint(shape.constraints['height'])
#     if np.random.uniform(0, 1) < epsilon:
#         shape.rad = np.random.randint(shape.constraints['width'])
#     if np.random.uniform(0, 1) < epsilon:
#         shape.alpha = float('{:.2f}'.format(np.random.uniform(0, 1)))
#     if np.random.uniform(0, 1) < epsilon:
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
T_max = 400
T_min = 1
delta_T = 0.9 # temperature decay
temps = np.arange(T_max, T_min, -delta_T)
theta = 0.2
all_losses = []
all_images = []

# augmented simulated annealing
for i in range(len(temps)):
    T = temps[i] # current temperature
   
    shape = Shape(constraints)
    
    canvas_with_shape = shape.superimpose(canvas)
    epsilon = get_score(target, canvas)
    epsilon_shape = get_score(target, canvas_with_shape)
    
    if epsilon_shape > epsilon > theta: # shape brings canvas closer to target
        canvas = canvas_with_shape
        epsilon = epsilon_shape
    elif np.exp(-epsilon / T) > np.random.uniform(0, 1):
        canvas = canvas_with_shape
        epsilon = epsilon_shape
        
    all_images.append(canvas)
    all_losses.append(epsilon)
    
plt.figure(1)
plt.subplot(121)
plt.imshow(all_images[-1])    
plt.subplot(122)
plt.imshow(canvas)
plt.show()
    
# animating image generation process
# fig = plt.figure()
# ax = plt.axes()
# line, = ax.plot([], [], lw=2)

# def init():
#     line.set_data([], [])
#     return line,

# for i in range(len(all_images)):
#     all_images[i] = [plt.imshow(all_images[i], animated=True)]
    
# anim = animation.ArtistAnimation(fig, all_images, blit=True, repeat_delay=5000)
# plt.show()

# plt.plot(temps, all_losses, color="green")
# plt.show()