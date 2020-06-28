import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shape import Shape

def read_image(path):
    img = Image.open(path)
    img = np.array(img)
    
    constraints = {}
    constraints['width'] = max(img.shape[:2])
    constraints['height'] = min(img.shape[:2])
    
    return img, constraints
    
def get_loss(og_img, generated):
    diff = np.square(og_img - generated)
    cost = np.sum(diff)
    cost = float(cost / (og_img.shape[0] * og_img.shape[1]))
    
    return cost
    
img, constraints = read_image("../assets/sky.jpg")
constraints['all_colours'] = list(set(tuple(v) for m2d in img for v in m2d))
average_colour = img.mean(axis=0).mean(axis=0)
canvas = np.full(shape=img.shape, fill_value=average_colour, dtype=np.int)
        
generations = 1000
n_shapes = 100

shapes = [Shape(constraints) for i in range(n_shapes)]

for gen in range(generations):
    for i in range(len(shapes)):
        cur_shape = shapes[i]
        
        img, gen_img = cur_shape.superimpose(img)
        
        