import numpy as np
import keras
from phobert_embeding import get_emb_vector
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, ids, labels, batch_size=16,  max_seq_len=256-2, feature_len=768, n_classes=18, shuffle=True):
        self.max_seq_len = max_seq_len
        self.feature_len = feature_len
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idx_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.max_seq_len, self.feature_len))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, idx in enumerate(idx_temp):
            X[i,] = get_emb_vector(self.ids[idx])
            # Store class
            y[i] = self.labels[idx]
        return X, y