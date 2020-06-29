import cv2, math, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import defaultdict
from PIL import Image
from shapes import Shape

def read_image(path):
    img = Image.open(path)
    
    constraints = {}
    constraints['width'] = max(np.array(img).shape[:2])
    constraints['height'] = min(np.array(img).shape[:2])
    
    return img, constraints
    
# standard MSE loss
def get_loss(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err
    
def mutate(shape):
    theta = 0.4
    if np.random.uniform(0, 1) < theta:
        shape.x = np.random.randint(shape.constraints['width'])
    if np.random.uniform(0, 1) < theta:
        shape.y = np.random.randint(shape.constraints['height'])
    if np.random.uniform(0, 1) < theta:
        shape.rad = np.random.randint(shape.constraints['width'])
    if np.random.uniform(0, 1) < theta:
        shape.alpha = float('{:.2f}'.format(np.random.uniform(0, 1)))
    if np.random.uniform(0, 1) < theta:
        shape.colour = random.choice(shape.constraints['all_colours'])
    
    return shape
    
img, constraints = read_image("../assets/sky.jpg")
constraints['all_colours'] = list(img.getdata())

img = np.array(img)
average_colour = img.mean(axis=0).mean(axis=0)
# canvas = np.full(shape=img.shape, fill_value=average_colour, dtype=np.int8)
canvas = np.zeros_like(img, dtype=np.uint8)
canvas[:] = 255
canvas = canvas.astype(np.int8)

# hyper params
count = 0
temperature = 200
temp_decay = 1
prev_loss = math.inf
all_losses = []
all_images = []

# simulated annealing
new_canvas = canvas
while temperature != 1:
    new_shape = Shape(constraints)
    
    new_canvas, canvas = new_shape.superimpose(new_canvas) # C -> N
    loss = get_loss(img, new_canvas) # ∆E
    
    # hill climbing
    new_mut_canvas = canvas
    if loss < prev_loss:
        mutated_shape = mutate(new_shape)
        new_mut_canvas, canvas = mutated_shape.superimpose(new_mut_canvas)
        mut_loss = get_loss(img, new_mut_canvas)
        
        if loss > mut_loss:
            canvas = new_canvas
            all_images.append(new_canvas)
            all_losses.append(loss)
        else:
            canvas = new_mut_canvas
            all_images.append(new_mut_canvas)
            all_losses.append(mut_loss)
            
    elif np.exp(-loss / temperature) > np.random.uniform(0, 1):
        canvas = new_canvas
        prev_loss = loss    
        
    temperature -= temp_decay
    count += 1

# plt.figure(1)
# plt.subplot(121)
# plt.imshow(canvas, cmap="gray", vmin=0, vmax=255)
# plt.subplot(122)
# plt.imshow(img)
# plt.show()

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

# plt.plot(range(count), all_losses, color="green")
# plt.show()