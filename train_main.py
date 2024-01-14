import json
import os

import pandas as pd
import numpy as np
import tensorflow as tf

# Below code prevents pre-allocating whole available GPU memory.
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for idx in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[idx], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# fp16
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# It is important to import model utils AFTER setting policy!

from utils_data_generator import get_tf_data_generator
from utils_model import MODEL, CustomSchedule, CustomNonPaddingSCCELabelSmoothLoss
from utils_callbacks import CustomCheckpointCallback, CustomTensorboardLoggerCallback


with open('config.json', 'r') as openfile:
    config = json.load(openfile)

SOURCE_LEN = config['SOURCE_LEN']
TARGET_LEN = config['TARGET_LEN']
VOCAB_SIZE = config['VOCAB_SIZE']
N = config['N']
H = config['H']
D_MODEL = config['D_MODEL']
DROPOUT = config['DROPOUT']
D_FF = D_MODEL * 4
D_KV = D_MODEL // H # This is default in PyTorch MHA and cannot be set explicitly.
LABEL_SMOOTHING = 0

# 26B tokens correspond to 100,000 steps on A100-40GB with batch_size = 1024 for XSmall model.
desired_token_exposure = 26_214_400_000
batch_size = 128
tf_ds = get_tf_data_generator(batch_size, num_parallel_calls = tf.data.AUTOTUNE)

weight_files = os.listdir('checkpoints')
weight_files = [file for file in weight_files if ".hdf5" in file]
if len(weight_files) > 0:
    last_epoch_was = pd.Series(weight_files).str.split('.').str[0].str.split('_').str[2].astype(int).max()
else:
    last_epoch_was = 0

epochs = 10
steps_per_epoch = int(desired_token_exposure / (batch_size * SOURCE_LEN * epochs))

learning_rate = CustomSchedule(D_MODEL, warmup_steps = 2_000, last_epoch = last_epoch_was, 
                               steps_per_epoch = steps_per_epoch)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-6)
loss = CustomNonPaddingSCCELabelSmoothLoss(LABEL_SMOOTHING, VOCAB_SIZE)

# Transformer Model
model = MODEL(N = N, SOURCE_LEN = SOURCE_LEN, TARGET_LEN = TARGET_LEN,
              VOCAB_SIZE = VOCAB_SIZE, D_MODEL = D_MODEL, D_FF = D_FF, D_KV = D_KV,
              H = H, DROPOUT = DROPOUT)
model.compile(optimizer = optimizer, loss = loss, jit_compile = True)


# Load the weights of the last checkpoint
if len(weight_files) > 0:
    idx_of_weight_to_load = pd.Series(weight_files).str.split('.').str[0].str.split('_').str[2].astype(int).idxmax()

    weight_to_load = weight_files[idx_of_weight_to_load]
    print('Loading checkpoint of last epoch:', weight_to_load)
    
    # Dummy forward pass to initiate the weights before loading
    X_enc = np.random.randint(low = 0, high = VOCAB_SIZE, size = (1, SOURCE_LEN))
    X_dec = np.random.randint(low = 0, high = VOCAB_SIZE, size = (1, TARGET_LEN))
    #with strategy.scope():
    _ = model((X_enc, X_dec))
    model.load_weights(f'checkpoints/{weight_to_load}')
else:
    last_epoch_was = 0

writer = tf.summary.create_file_writer("tensorboard_logs")
tensorboard = CustomTensorboardLoggerCallback(writer)

checkpoint = CustomCheckpointCallback()
model.fit(tf_ds, epochs = epochs, steps_per_epoch = steps_per_epoch, initial_epoch = last_epoch_was, verbose = 1,
          callbacks = [checkpoint, tensorboard])