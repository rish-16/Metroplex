import numpy as np
import cv2
import random

class Shape:
    def __init__(self, constraints):
        self.colour = random.choice(constraints['all_colours'])
        self.x1 = np.random.randint(constraints['width'])
        self.y1 = np.random.randint(constraints['height'])
        self.x2 = np.random.randint(constraints['width'])
        self.y2 = np.random.randint(constraints['height'])
        self.alpha = float('{:.2f}'.format(np.random.uniform(0, 1)))
        
    def superimpose(self, og_img):
        blk = np.zeros(og_img.shape, dtype=np.int8)
        cv2.rectangle(blk, (self.x1, self.y1), (self.x2, self.y2), self.colour, cv2.FILLED)
        new_img = cv2.addWeighted(og_img, self.alpha, blk, 1-self.alpha, 0)
        
        return new_img, og_img
        
    def describe(self):
        print (self.__dict__)
        
class MutShape(Shape):
    def __init__(self, constraints):
        super().__init__(constraints)
        
        