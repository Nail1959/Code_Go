#-*- coding:UTF-8 -*-
# Генерация данных для обучения и тестирования.
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
import os
import fnmatch

num_games = int(input('Количество игр для обучения и тестирования ='))
seed = 1377
data_dir = r'/home/nail/Code_Go/dlgo/data'
encoder = SimpleEncoder((19,19))

lst_files = os.listdir(data_dir)
lst_npy=[]
for entry in lst_files:
   if fnmatch.fnmatch(entry, '*npy'):
      lst_npy.append(entry)
cnt_files_begin = len(lst_npy)

print('---------------------------  Count files npy Before = ', cnt_files_begin)

processor = GoDataProcessor(encoder=encoder.name(), data_directory='data')

generator = processor.load_go_data('train', num_games, use_generator=True, seed = seed)
test_generator = processor.load_go_data('test', num_games, use_generator=True,seed = seed)

lst_files = os.listdir(data_dir)
lst_npy=[]
for entry in lst_files:
   if fnmatch.fnmatch(entry, '*npy'):
      lst_npy.append(entry)

cnt_files_after = len(lst_npy)
print('----------------------------  Count files npy Before = ', cnt_files_begin)
print('============================   Count files npy After = ', cnt_files_after)
print('---- ------------------------------------------------------New files = ',
      cnt_files_after - cnt_files_begin)
print('Only formed KGS files')