import numpy as np
import dataParse

x_path = '../data/narray/x.npy'
y_path = '../data/narray/y.npy'
z_path = '../data/narray/z.npy'
w_path = '../data/narray/w.npy'
x_transform_path = '../data/narray/x_transform.npy'
y_transform_path = '../data/narray/y_transform.npy'
alpha_path = '../data/narray/alpha.npy'
beta_path = '../data/narray/beta.npy'
inputs_path = '../data/narray/inputs.npy'
outputs_path = '../data/narray/outputs.npy'
x_mean_path = '../data/narray/x_mean.npy'
y_mean_path = '../data/narray/y_mean.npy'


def my_pca(data, k):
    (row, col) = data.shape
    mean_data1 = np.mean(data, axis=0)
    mean_data = np.tile(mean_data1, (row, 1))
    center_data = data-mean_data
    # np.cov see row as a feature
    cov_data = np.cov(center_data.T)
    (eigs, vec) = np.linalg.eig(cov_data)

    sorted_indices = np.argsort(eigs)
    top_k_vec = vec[:, sorted_indices[:-k-1:-1]]
    pca_data = np.matmul(center_data, top_k_vec)
    # recovery3 = np.matmul(pca_data, top_k_vec.T)+ mean_data

    return pca_data, top_k_vec, mean_data1


# x.shape = (n, 3c)   y.shape = (n, 3)
# save data as one file to acc data reading
# bunny_path = '../data/simdata/'
# sphere_path = '../data/spherecenter/spherecenter.txt'
# x = dataParse.data_parse_bunny(bunny_path)
# np.save(x_path, x)
# y = dataParse.data_parse_sphere(sphere_path)
# np.save(y_path, y)
# exit(-1)


x = np.load(x_path)
y = np.load(y_path)
# print(x.shape)
# print(y.shape)


# x: bunny = (3c * n), y:sphere = (3 * n)
num_pca_x = 64      # bunny
num_pca_y = 3       # sphere, which means no PCA applied

# z = (n, num_pca_x)  indicate bunny after pca_x
# w = (n, num_pca_y)  indicate sphere after pca_y
(z, x_transform, x_mean) = my_pca(x, num_pca_x)
(w, y_transform, y_mean) = my_pca(y, num_pca_y)

# np.save(x_mean_path, x_mean)
# np.save(y_mean_path, y_mean)
# np.save(z_path, z)
# np.save(w_path, w)
# np.save(x_transform_path, x_transform)
# np.save(x_transform_path, x_transform)
#
# exit(0)

#  compute alpha, beta by solving a linear least squares
# construct Ax = B and argmin ||b-Ax||^2
# Zt = alpha * Zt-1 + beta * (Zt-1 - Zt-2)
alpha = np.zeros(num_pca_x)
beta = np.zeros(num_pca_x)

for i in range(0, num_pca_x):
    (rows, cols) = z.shape
    cur_z = z[:, i]
    b = cur_z[2:rows]
    A = np.zeros([rows-2, 2])
    A[:, 0] = cur_z[1:rows-1]

    for j in range(1, rows-1):
        A[j-1, 1] = cur_z[j] - cur_z[j-1]

    res = np.linalg.lstsq(A, b, rcond=None)
    alpha[i] = res[0][0]
    beta[i] = res[0][1]


np.save(alpha_path, alpha)
np.save(beta_path, beta)
print(alpha)
print(beta)
[rows, cols] = z.shape
z_init = np.zeros((rows-2, cols))

# compute z_init, attention
for i in range(2, rows):
    z_init[i-2, :] = alpha * z[i-1, :] + beta * (z[i-1, :] - z[i-2, :])

# delta_z = z - z_init and it's our target: (n-2, num_pca_x)
delta_z = z[2:rows, :] - z_init

# input data [/zt;zt-1;wt], handle bunny and sphere firstly
inputs = np.hstack((z_init, z[1:rows-1, :], w[2:rows, :]))
# network outputs
outputs = delta_z
# print(inputs.shape)
# print(outputs.shape)

np.save(inputs_path, inputs)
np.save(outputs_path, outputs)



