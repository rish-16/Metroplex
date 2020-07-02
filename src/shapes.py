import numpy as np
import cv2, random

class Shape:
    def __init__(self, constraints):
        self.constraints = constraints
        self.colour = random.choice(constraints['all_colours'])
        self.rad = np.random.randint(100)
        self.x = np.random.randint(constraints['width'])
        self.y = np.random.randint(constraints['height'])
        self.alpha = float('{:.2f}'.format(np.random.choice(np.arange(0, 1, 0.05))))
        
    def superimpose(self, og_img):
        img_copy = og_img.copy()
        # blk = np.zeros(img_copy.shape, dtype=np.int8)
        # cv2.circle(blk, (self.x, self.y), self.rad, self.colour, cv2.FILLED)
        # new_img = cv2.addWeighted(img_copy, self.alpha, blk, 1-self.alpha, 0)
        new_img = cv2.circle(img_copy, (self.x, self.y), self.rad, self.colour, cv2.FILLED)
        
        return new_img
        
    def describe(self):
        print (self.rad, self.colour, self.alpha, (self.x, self.y))