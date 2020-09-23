import os
import fnmatch

data_dir = '//home//nail//Code_Go//dlgo//data'
lst_files = os.listdir(data_dir)
lst_npy=[]
for entry in lst_files:
   if fnmatch.fnmatch(entry, '*train*npy'):
      lst_npy.append(entry)
cnt_files = len(lst_npy)

print('Количество файлов в директории '+data_dir+' = ',cnt_files)