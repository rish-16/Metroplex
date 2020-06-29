import numpy as np
import cv2, random
from skimage.measure import compare_ssim

class Shape:
    def __init__(self, constraints):
        self.constraints = constraints
        self.colour = random.choice(constraints['all_colours'])
        self.rad = np.random.randint(50)
        self.x = np.random.randint(constraints['width'])
        self.y = np.random.randint(constraints['height'])
        self.alpha = float('{:.2f}'.format(np.random.uniform(0, 1)))
        
    def superimpose(self, og_img):
        blk = np.zeros(og_img.shape, dtype=np.int8)
        cv2.circle(blk, (self.x, self.y), self.rad, self.colour, cv2.FILLED)
        new_img = cv2.addWeighted(og_img, self.alpha, blk, 1-self.alpha, 0)
        
        return new_img, og_img
        
    def describe(self):
        print (self.rad, self.colour, self.alpha, (self.x, self.y))
        
    # standard Structural Similarity Index (SSIM) loss
    def get_score(self, og_img, gen_img):
        (score, diff) = compare_ssim(og_img, gen_img, full=True, multichannel=True)
        
        return score