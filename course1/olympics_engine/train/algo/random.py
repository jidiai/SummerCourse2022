import random

class random_agent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        #self.seed(seed)

    def seed(self, seed = None):
        random.seed(seed)

    def act(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [[force], [angle]]

