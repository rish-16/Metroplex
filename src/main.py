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
    
def nrmse(og_img, gen_img):
    mse = np.sqrt(np.mean(np.square(og_img - gen_img)))
    err = mse / (og_img.max() - og_img.min())
    
    return err

def read_image(path):
    img = Image.open(path)
    
    constraints = {}
    constraints['width'] = max(np.array(img).shape[:2])
    constraints['height'] = min(np.array(img).shape[:2])
    
    return img, constraints
    
def mutate(shape):
    gamma = 0.4
    if np.random.uniform(0, 1) < gamma:
        shape.x = np.random.randint(shape.constraints['width'])
    if np.random.uniform(0, 1) < gamma:
        shape.y = np.random.randint(shape.constraints['height'])
    if np.random.uniform(0, 1) < gamma:
        shape.rad = np.random.randint(75)
    if np.random.uniform(0, 1) < gamma:
        shape.alpha = float('{:.2f}'.format(np.random.choice(np.arange(0, 1, 0.05))))
    if np.random.uniform(0, 1) < gamma:
        shape.colour = random.choice(shape.constraints['all_colours'])
    
    return shape
    
target, constraints = read_image("../assets/face.jpg")
constraints['all_colours'] = list(target.getdata())

target = np.array(target)
canvas = np.zeros_like(target)
canvas[:] = 255

# hyper params
T_max = 500
T_min = 1
delta_T = 0.9 # temperature decay
temps = np.arange(T_max, T_min, -delta_T)
theta = 0.4
all_losses = []
all_images = []

# simulated annealing
for i in range(len(temps)):
    T = temps[i] # current temperature
   
    shape = Shape(constraints)
    
    canvas_with_shape = shape.superimpose(canvas)
    epsilon = get_score(target, canvas)
    epsilon_shape = get_score(target, canvas_with_shape)
    
    print (nrmse(target, canvas), nrmse(target, canvas_with_shape))
    
    if epsilon_shape > epsilon > theta: # shape brings canvas closer to target
        
        # hill climbing
        mut_shape = mutate(shape)
        canvas_mut_shape = mut_shape.superimpose(canvas)
        epsilon_mut = get_score(target, canvas_mut_shape)
        
        if epsilon_mut > epsilon_shape: # mutated shape is better than original shape
            canvas = canvas_mut_shape
            epsilon = epsilon_mut
        else:
            canvas = canvas_with_shape
            epsilon = epsilon_shape
            
    elif np.exp(-epsilon / T) > np.random.uniform(0, 1):
        canvas = canvas_with_shape
        epsilon = epsilon_shape
        
    all_images.append(canvas)
    all_losses.append(epsilon)
    
plt.figure(1)
plt.subplot(121)
plt.imshow(canvas)
plt.subplot(122)
plt.imshow(target)
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