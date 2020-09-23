import glob
import numpy as np
from  dlgo.data.generator import DataGenerator
from dlgo.data.sampling import Sampler

data_dir = r'/home/nail/Code_Go/dlgo/data'

num_games = 20000

sampler = Sampler(data_dir=data_dir, seed=0)
data = sampler.draw_data('train', num_games)
generator = DataGenerator(data_dir, data)

num_samples = 0
batch_size = 256
num_classes = 19*19
for X, y in generator._generate(batch_size=batch_size, num_classes=num_classes):
    num_samples += X.shape[0]

print('Num_samples = ', num_samples)
