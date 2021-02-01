import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import Input
from keras.layers import Flatten, Conv2D, MaxPool2D, Conv1D, GlobalMaxPooling1D, Dropout, Dense, BatchNormalization
from keras.utils import multi_gpu_model

from phobert_embeding import text2ids
from dataloader import DataGenerator

classes  = ['__label__sống_trẻ', '__label__thời_sự', '__label__công_nghệ', '__label__sức_khỏe', '__label__giáo_dục', '__label__xe_360', '__label__thời_trang', '__label__du_lịch', '__label__âm_nhạc', '__label__xuất_bản', '__label__nhịp_sống', '__label__kinh_doanh', '__label__pháp_luật', '__label__ẩm_thực', '__label__thế_giới', '__label__thể_thao', '__label__giải_trí', '__label__phim_ảnh']
print("classes:", classes)

# def load_text(filename):
#     labels, ids = [],[]
#     with open(filename, 'r', encoding="utf-8") as f:
#         c = 0
#         for line in f:
#             print(c)
#             line = line.strip().lower().split(" ",1)
#             label = line[0]
#             text = line[1]
#             # print(label)
#             tkz = text2ids(text)
#             ids.append(tkz)
#             labels.append(classes.index(label))
#             c += 1
#     ids = tf.keras.preprocessing.sequence.pad_sequences(ids, maxlen=256, dtype="int32", value=0, truncating="post", padding="post")
#     ids[:,-1] = 2
#     # print(ids[0])
#     return ids, labels

# ids_train, y_train = load_text("train.txt")
# ids_test, y_test = load_text("test.txt")

# print(len(ids_train))
# print(len(y_train))
# print(len(ids_test))
# print(len(y_test))
# # exit()
# with open("ids_train", 'wb') as f:
#     pickle.dump(ids_train, f)
# with open("y_train", 'wb') as f:
#     pickle.dump(y_train, f)

# with open("ids_test", 'wb') as f:
#     pickle.dump(ids_test, f)
# with open("y_test", 'wb') as f:
#     pickle.dump(y_test, f)

# exit()
# print("LOAD DATA DONE")

with open("ids_train", 'rb') as f:
    ids_train = pickle.load(f)
with open("y_train", 'rb') as f:
    y_train = pickle.load(f)
with open("ids_test", 'rb') as f:
    ids_test = pickle.load(f)
with open("y_test", 'rb') as f:
    y_test = pickle.load(f)

print(ids_train)

print("LOAD DATA FROM FILE DONE")

# exit()
train_generator = DataGenerator(ids_train, y_train, batch_size=8, n_classes=len(classes))
valid_generator = DataGenerator(ids_test, y_test, batch_size=8, n_classes=len(classes), shuffle=False)

print("train_generator:",len(train_generator))
print("valid_generator:",len(valid_generator))

model = Sequential()
model.add(Conv1D(128, kernel_size=5, input_shape=(256,768)))
# model.add(GlobalMaxPooling1D())
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
# model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# model = Sequential()
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(len(classes), activation='softmax'))

# model=multi_gpu_model(model, gpus=2)

# define the model
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(256,768)),
#     tf.compat.v1.keras.layers.CuDNNLSTM(64,  return_sequences=False),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(len(classes))
# ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# fit the model
model.fit_generator(train_generator, epochs=10, verbose=1, validation_data=valid_generator)
# evaluate the model
loss, accuracy = model.evaluate_generator(valid_generator, verbose=1)
print('Accuracy: %f' % (accuracy*100))

