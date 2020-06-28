import numpy as np
import cv2

class Shape:
    def __init__(self, constraints):
        self.epsilon = 0
        self.colour = np.random.choice(constraints['all_colours'])
        self.x1 = np.random.randint(0, constraints['width'])
        self.y1 = np.random.randint(0, constraints['height'])
        self.x2 = np.random.randint(0, constraints['width'])
        self.y2 = np.random.randint(0, constraints['height'])
        self.rotation = np.random.choice([0, 45, 90])
        
    def superimpose(self, img):
        new_img = cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), self.colour, -1)
        
        return new_img, img
        
    def get_score(self, og, gen, gen_with_shape):
        
        
    