import tensorflow as tf
import numpy as np
from model import build_model
import dataPrepcocess
import utility
import re
import os

#  need to handle  num_pca_x, cause it appeared in train.py
num_pca_x = dataPrepcocess.num_pca_x    # bunny
num_pca_y = dataPrepcocess.num_pca_y    # sphere
z_path = dataPrepcocess.z_path
w_path = dataPrepcocess.w_path
alpha_path = dataPrepcocess.alpha_path
beta_path = dataPrepcocess.beta_path
x_transform_path = dataPrepcocess.x_transform_path
x_mean_path = dataPrepcocess.x_mean_path


checkpoint_dir = 'checkpoint'
latest = tf.train.latest_checkpoint(checkpoint_dir)
print('load paras: ', latest)

model = build_model(num_pca_x, num_pca_x*2 + num_pca_y)
model.load_weights(latest)

z = np.load(z_path)
w = np.load(w_path)
alpha = np.load(alpha_path)
beta = np.load(beta_path)

# select t=2, make sure t >= 2
# input data for model [/zt;zt-1;wt]
t = 773+2
# /zt = alpha * zt-1 + beta * (zt-1 - zt-2)
z_init = alpha * z[t-1, :] + beta * (z[t-1, :] - z[t-2, :])
input_data = np.hstack((z_init, z[t-1, :], w[t, :]))
input_batch = np.array([input_data])
result = model.predict(input_batch)


predict = z_init + result
# convert predict to real vertices(3c), got the matrix U(with u_transpose * predict)
x_transform_mat = np.load(x_transform_path)
x_mean = np.load(x_mean_path)
x_recovery = np.matmul(predict, x_transform_mat.T) + x_mean


dir = './data/results/' + str(t+1) + '/'
if not os.path.exists(dir):
    os.makedirs(dir)


bunny_save_path = dir + 'bunny.obj'
bunny_truth_save_path = dir + 'bunnytruth.obj'
sphere_save_path = dir + 'sphere.obj'
bunny_face_path = './data/obj/simdata.obj'
sphere_face_path = './data/obj/simdatacollision.obj'



# now x_recovery indicate x for frame t, w[t, :] indicates sphere at frame t, save as .obj file
# save bunny
utility.save_obj_vertices(bunny_save_path, x_recovery[0, :])
bunny_faces = utility.parse_face(bunny_face_path)
utility.append_face(bunny_save_path, bunny_faces)

bunny_ground_truth_vertices_path = './data/simdata/' + 'softbody' + str(t+1) + '.data'
with open(bunny_ground_truth_vertices_path) as file:
    line = file.readline()
    arr = np.vstack([np.fromstring(i, sep=', ') for i in re.findall('\[(.+?)\]', line)])
    utility.save_obj_vertices(bunny_truth_save_path, arr.flatten())
utility.append_face(bunny_truth_save_path, bunny_faces)


# save sphere for file order = t+1
sphere_vertices_path = './data/simdatacollision/' + 'softbody' + str(t+1) + 'collision.data'
with open(sphere_vertices_path) as file:
    line = file.readline()
    arr = np.vstack([np.fromstring(i, sep=', ') for i in re.findall('\[(.+?)\]', line)])
    utility.save_obj_vertices(sphere_save_path, arr.flatten())

sphere_faces = utility.parse_face(sphere_face_path)
utility.append_face(sphere_save_path, sphere_faces)
