import cv2, math, random, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapes import Shape

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input image path")
parser.add_argument("-o", "--output", help="Output image path")
parser.add_argument("--solo", help="Whether output should be side-by-side comparison or generated image only", action="store_true")
args = parser.parse_args()
    
def get_op_filename(filename):
    path = filename.split(".")
    ext = path[-1]
    filepath = ".".join(path[:-1]) + "_output."
    return  filepath + ext
    
if args.input == False:
    raise FileNotFoundError("Input filepath required.")
else:
    filename = args.input
    
    if args.output == False:
        op_filename = ""
    else:
        op_filename = get_op_filename(filename)

# normalized to [0, 1]
def nrmse(og_img, gen_img):
    mse = np.sqrt(np.mean(np.square(og_img - gen_img)))
    err = mse / (og_img.max() - og_img.min())
    
    return err

def read_image(path):
    img = Image.open(path)
    print ("Loaded image at '{}'".format(path))
    
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
        shape.rad = np.random.randint(50)
    if np.random.uniform(0, 1) < gamma:
        shape.alpha = float('{:.2f}'.format(np.random.choice(np.arange(0, 1, 0.05))))
    if np.random.uniform(0, 1) < gamma:
        shape.colour = random.choice(shape.constraints['all_colours'])
    
    return shape
    
target, constraints = read_image(filename)
constraints['all_colours'] = list(target.getdata())

target = np.array(target)
canvas = np.zeros_like(target)
canvas[:] = 255

# hyper params
T_max = 1000
T_min = 1
T_delta = 0.05 # temperature decay
temps = np.arange(T_max, T_min, -T_delta)
theta = 1
all_losses = []
all_images = []

# simulated annealing
print ("Generating image...")
for i in range(len(temps)):
    T = temps[i] # current temperature
   
    shape = Shape(constraints)
    
    canvas_with_shape = shape.superimpose(canvas)
    epsilon = nrmse(target, canvas)
    epsilon_shape = nrmse(target, canvas_with_shape)
    
    if epsilon_shape < epsilon < theta: # shape brings canvas closer to target
        
        # hill climbing
        mut_shape = mutate(shape)
        canvas_mut_shape = mut_shape.superimpose(canvas)
        epsilon_mut = nrmse(target, canvas_mut_shape)
        
        if epsilon_mut < epsilon_shape: # mutated shape is better than original shape
            canvas = canvas_mut_shape
            epsilon = epsilon_mut
        else:
            canvas = canvas_with_shape
            epsilon = epsilon_shape
            
    elif np.exp(-epsilon / T) < np.random.uniform(0, 1): # not using Boltzmann Constant
        canvas = canvas_with_shape
        epsilon = epsilon_shape
        
    all_images.append(canvas)
    all_losses.append(epsilon)
    
if args.solo == True: # generated image only    
    plt.tight_layout()
    plt.imshow(canvas)
    plt.axis("off")
else:
    plt.tight_layout()
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(canvas)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(target)
    plt.axis("off")
    
plt.savefig(op_filename)
print ("Image saved at '{}'".format(op_filename))