import glob
import numpy as np
from  dlgo.data.generator import DataGenerator
from dlgo.data.sampling import Sampler

data_dir = r'/home/nail/Code_Go/dlgo/data'

num_games = 20000

sampler = Sampler(data_dir=data_dir, seed=0)
data = sampler.draw_data('train', num_games)
generator = DataGenerator(data_dir, data)

data = sampler.draw_data('test', num_games)
test_generator = DataGenerator(data_dir, data)

num_samples = 0
batch_size = 256
num_classes = 19*19
i = 0
for X, y in generator._generate(batch_size=batch_size, num_classes=num_classes):
    num_samples += X.shape[0]
    i += 1
    print('Train I = ',i, ' Num_samples = ', num_samples)



sampler = Sampler(data_dir=data_dir, seed=0)
data = sampler.draw_data('test', num_games)
test_generator = DataGenerator(data_dir, data)

num_samples_test = 0
i = 0
for X, y in test_generator._generate(batch_size=batch_size, num_classes=num_classes):
    num_samples_test += X.shape[0]
    i += 1
    print('Test I = ', i, ' Num_samples = ', num_samples_test)

print('Train num_samples = ', num_samples)
print('Test num_samples = ', num_samples_test)
fname_samples = data_dir+'//file_num_samples.txt'

fn_samples = open(fname_samples,'w')
fn_samples.write(str(num_samples))
fn_samples.write(str(num_samples_test))
fname_samples.close()