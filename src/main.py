import cv2, math, random
import numpy as np
import matplotlib.pyplot as plt
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
canvas = np.full(shape=img.shape, fill_value=average_colour, dtype=np.int8)
# canvas = np.zeros_like(img, dtype=np.uint8)
# canvas[:] = 255
# canvas = canvas.astype(np.int8)

# hyper params
count = 0
temperature = 200
temp_decay = 1
prev_score = math.inf
all_losses = []

# simulated annealing
while temperature != 1:
    new_shape = Shape(constraints)
    
    new_canvas, canvas = new_shape.superimpose(canvas) # C -> N
    score = get_loss(img, new_canvas) # ∆E
    
    # hill climbing
    if score < prev_score:
        new_shape.describe()
        mutated_shape = mutate(new_shape)
        new_mut_canvas, canvas = mutated_shape.superimpose(canvas)
        mut_score = get_loss(img, new_mut_canvas)
        
        if score > mut_score:
            canvas = new_canvas
            mutated_shape.describe()
            all_losses.append(score)
        else:
            canvas = new_mut_canvas
            all_losses.append(mut_score)
            
    elif np.exp(-score / temperature) > np.random.uniform(0, 1):
        canvas = new_canvas
        prev_score = score    
        
    temperature -= temp_decay
    count += 1

plt.figure(1)
plt.subplot(121)
plt.imshow(canvas, cmap="gray", vmin=0, vmax=255)
plt.subplot(122)
plt.imshow(img)
plt.show()

plt.plot(range(count), all_losses, color="green")
plt.show()