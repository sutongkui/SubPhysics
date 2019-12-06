import numpy as np

# data format [x0,y0,z0,x1,y1,z1 ...]


def save_obj_vertices(path, data):
    f = open(path, 'w')
    print(len(data))
    for i in range(0, int(len(data) / 3)):
        f.write('v ' + str(data[i*3+0]) + ' ' + str(data[i*3+1]) + ' ' + str(data[i*3+2]) + '\n')
    f.close()

# data format [f0_0,f0_1,f0_2,f1_0,f1_1,f1_2 ...]


def append_face(path, data):
    f = open(path, 'a')
    for i in range(0, int(len(data) / 3)):
        f.write('f ' + str(data[i*3+0]) + ' ' + str(data[i*3+1]) + ' ' + str(data[i*3+2]) + '\n')
    f.close()


def parse_face(path):
    f = open(path)
    faces = []
    for line in f:
        values = line.split()
        if values[0] == 'f':
            for x in values[1:4]:
                faces.append(int(x))
    f.close()
    return np.array(faces)