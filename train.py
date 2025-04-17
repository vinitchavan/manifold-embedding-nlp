# train_model.py
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
import numpy as np

def triplet_loss(margin=1.0):
    def _loss(y_true, y_pred):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return loss
    return _loss

def train_triplet_model(model, anchor_input, pos_input, neg_input, batch_size=128, epochs=10):
    dummy_y = np.zeros((anchor_input.shape[0],))
    model.compile(optimizer=Adam(), loss=triplet_loss())
    model.fit(
        [anchor_input, pos_input, neg_input],
        dummy_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
