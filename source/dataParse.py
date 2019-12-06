import numpy as np
import re
import os
import sys

# read file from frame1 to frame 20000 (python read it randomly)
# bunny data format [1,1,1][2,2,2]...


def data_parse_bunny(file_path, num_read_file=sys.maxsize):
    all_frames_points = np.array([])
    first_file = True
    dirs = os.listdir(file_path)
    dirs.sort(key=lambda f: int(re.sub("\D", "", f)))
    num_real_file = len(dirs)

    num_need_to_read = min(num_read_file, num_real_file)
    i = 0
    for filename in dirs:
        # read vertex only
        with open(os.path.join(file_path, filename)) as file:
            line = file.readline()
            arr = np.vstack([np.fromstring(i, sep=', ') for i in re.findall('\[(.+?)\]', line)])
            # (num_vertices, 3) ---> (num_vertices * 3, ) order: x, y, z
            arr = arr.flatten()
            if first_file:
                all_frames_points = arr
                first_file = False
            else:
                all_frames_points = np.vstack((all_frames_points, arr))

        i = i+1
        if i == num_need_to_read:
            break

    return all_frames_points


# sphere center data format np.savetxt
def data_parse_sphere(file_path):
    ret = []
    with open(file_path) as f:
        for line in f:
            str_line = line.split()
            pos = [str_line[0], str_line[1], str_line[2]]
            ret.append(pos)

    return np.array(ret) 

# test = data_parse_sphere('../data/spherecenter/spherecenter.txt')
# print(test.shape)










