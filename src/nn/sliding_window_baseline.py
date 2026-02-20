from collections import deque
import random

class SlidingWindowBinaryClassification:
    def __init__(self, window_size = 7):
        self.mem = deque(maxlen=window_size)
    
    def update(self, x:int):
        self.mem.append(x)
    
    def __call__(self):
        if len(self.mem) == 0 :
            return random.randint(0, 1)
        return int(sum(self.mem)/len(self.mem) > 0.5) # Take majority vote
    
class SlidingWindowRegression:
    def __init__(self, window_size = 7):
        self.mem = deque(maxlen=window_size)

    def update(self, x:float):
        self.mem.append(x)

    def __call__(self):
        return sum(self.mem)/len(self.mem) 

