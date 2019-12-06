import tensorflow as tf

import numpy as np
from model import build_model

import dataPrepcocess

#  Limit gpu memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# set paras
EPOCHS = 100
batch_size = 160
# window_size must > 2
window_size = 3
delta_t = 1
train_ratio = 0.8


# Start training
z_path = dataPrepcocess.z_path
w_path = dataPrepcocess.w_path
alpha_path = dataPrepcocess.alpha_path
beta_path = dataPrepcocess.beta_path
num_pca_x = dataPrepcocess.num_pca_x    # bunny
num_pca_y = dataPrepcocess.num_pca_y    # sphere

all_z = np.load(z_path)
all_w = np.load(w_path)
alpha = np.load(alpha_path)
beta = np.load(beta_path)

# make sure consecutive for training
(all_z_row, all_z_col) = all_z.shape
num_z_train = int(all_z_row * train_ratio)
z_train = all_z[0:num_z_train, :]
z_test = all_z[num_z_train:, :]
w_train = all_w[0:num_z_train, :]
w_test = all_w[num_z_train:, :]

# input data [/zt;zt-1;wt], handle bunny and sphere firstly
model = build_model(num_pca_x, num_pca_x*2 + num_pca_y)
optimizer = tf.optimizers.Adam(0.0001, 0.999)


# num_window = window_size - 2
def construct_sample(input_z, input_w, idx):
    (input_z_row, input_z_col) = input_z.shape
    (input_w_row, input_w_col) = input_w.shape
    if idx + 2 >= input_z_row:
        return np.array([]), np.array([]), np.array([]), np.array([])

    z0 = input_z[idx, :]
    z1 = input_z[idx+1, :]

    noise0 = np.random.normal(0, 0.01, input_z_col)
    noise1 = np.random.normal(0, 0.01, input_z_col)

    z0 = z0 + noise0
    z1 = z1 + noise1

    num_window = min(window_size, input_z_row-idx-1)
    z_window_input = np.zeros((num_window-2, input_z_col*2 + input_w_col))
    z_window_predict = np.zeros((num_window-2, input_z_col))
    for k in range(2, num_window):
        z_init = alpha * z1 + beta * (z1 - z0)
        input_data = np.hstack((z_init, z1, input_w[idx + k, :]))
        input_batch1 = np.array([input_data])
        result = model.predict(input_batch1)
        predict = z_init + result[0, :]

        z_window_input[k-2, :] = input_data
        z_window_predict[k-2, :] = predict

        z0 = z1
        z1 = predict

    # compute loss with mean absolute error
    z_window_truth = input_z[idx+2:idx+num_window, :]
    (z_window_rows, z_window_cols) = z_window_truth.shape

    # compute velocity
    z_window_last_truth = np.vstack((input_z[idx + 1, :], z_window_truth[0:z_window_rows - 1, :]))
    z_window_last_predict = np.vstack((z1, z_window_predict[0:z_window_rows - 1, :]))

    return z_window_input, z_window_truth, z_window_last_truth, z_window_last_predict


def construct_batch(z_input, w_input, start, end):
    batch_input = np.array([])
    output_truth = np.array([])
    last_truth = np.array([])
    last_predict = np.array([])
    for j in range(start, end):
        if j == start:
            batch_input, output_truth, last_truth, last_predict = construct_sample(z_input, w_input, j)
        else:
            tem_input, tem_truth, tem_last_truth, tem_last_predict = construct_sample(z_input, w_input, j)
            if tem_input.size:
                batch_input = np.vstack((batch_input, tem_input))
                output_truth = np.vstack((output_truth, tem_truth))
                last_truth = np.vstack((last_truth, tem_last_truth))
                last_predict = np.vstack((last_predict, tem_last_predict))

    return batch_input, output_truth, last_truth, last_predict


def loss_fn(predict, last_predict,  truth,  last_truth):
    predict = tf.dtypes.cast(predict, tf.float64)
    truth = tf.dtypes.cast(truth, tf.float64)
    last_truth = tf.dtypes.cast(last_truth, tf.float64)
    last_predict = tf.dtypes.cast(last_predict, tf.float64)

    delta_abs = tf.abs(tf.abs(predict) - tf.abs(truth))
    pos_loss = tf.reduce_mean(delta_abs)

    # compute velocity
    vel_predict = (predict - last_predict) / delta_t
    vel_truth = (truth - last_truth) / delta_t

    delta_abs_vel = tf.abs(tf.abs(vel_predict) - tf.abs(vel_truth))
    vel_loss = tf.reduce_mean(delta_abs_vel)
    # print('pos_loss ', pos_loss)
    # print('vel_loss ', vel_loss)

    return pos_loss + vel_loss


(z_rows, z_cols) = z_train.shape

logdir = "./logs"
writer = tf.summary.create_file_writer(logdir)

with writer.as_default():

    # Iterate over epochs.
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,))
        step = 0
        for i in range(0, z_rows, batch_size):
            # skip the final batch whose size is less than batch_size
            if i + batch_size > z_rows:
                break

            batch_start = i
            batch_end = min(batch_start + batch_size, z_rows)
            step += 1

            # construct batch input: batch_size * (window_size-2)
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            input_batch, batch_truth, batch_last_truth, batch_last_predict = construct_batch(z_train, w_train, batch_start, batch_end)
            with tf.GradientTape() as tape:
                batch_prediction = model(tf.convert_to_tensor(input_batch, dtype=np.float32))
                batch_loss = loss_fn(batch_prediction, batch_last_predict, batch_truth, batch_last_truth)


            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(batch_loss, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 100 == 0:
                print('Training loss (for one batch) at step %s: %s' % (epoch*EPOCHS+step, float(batch_loss)))
                tf.summary.scalar('loss', float(batch_loss), step=epoch*EPOCHS+step)


# Save entire model to a HDF5 file
model.save('SavedModel/my_model_tf13.h5')