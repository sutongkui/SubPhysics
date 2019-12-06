import numpy as np
import re
import os

# compute sphere center coordinate and save it as a file
file_path = '../data/simdatacollision/'
savePath = '../data/spherecenter/spherecenter.txt'

dirs = os.listdir(file_path)
dirs.sort(key=lambda f: int(re.sub("\D", "", f)))

with open(savePath, 'w') as savefile:
    for filename in dirs:
        print(filename)
        # read vertex only
        with open(os.path.join(file_path, filename)) as file:
            line = file.readline()
            arr = np.vstack([np.fromstring(i, sep=', ') for i in re.findall('\[(.+?)\]', line)])
            (rows, cols) = arr.shape
            center = np.zeros(3)
            for i in range(0, rows):
                center = center + arr[i, :] / rows
            
            list_center = center.tolist()  
            str_center = ''
            for x in list_center:
                str_center = str_center + ' ' + '{:.7f}'.format(x)        
            str_center += '\n'  
            savefile.write(str_center)
