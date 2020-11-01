import time

class Timer(object):
    def __init__(self):
        self.tick()


    def tick(self):
        self.start_time = time.time()


    def tock(self):
        self.end_time = time.time()
        time_since_tick = self.end_time - self.start_time
        self.tick()
        return time_since_tick

