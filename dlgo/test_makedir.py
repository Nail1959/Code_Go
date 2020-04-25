import os
import sys
name=input('Path and name file: ')
os.makedirs(name, mode=0o777, exist_ok=True)