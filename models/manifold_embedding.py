
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ManifoldEmbedding(tf.keras.Model):
    def __init__(self, vocab_size, max_len, emb_dim=64, projection_type='sphere'):
        super(ManifoldEmbedding, self).__init__()
        self.embedding = layers.Embedding(vocab_size, emb_dim, input_length=max_len)
        self.pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(3)
        self.projection_type = projection_type

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.pooling(x)
        x = self.dense(x)
        return self.project(x)

    def project(self, x):
        if self.projection_type == 'sphere':
            return self.project_to_sphere(x)
        elif self.projection_type == 'torus':
            return self.project_to_torus(x)
        elif self.projection_type == 'mobius':
            return self.project_to_mobius(x)
        else:
            raise ValueError("Unknown projection type: choose from 'sphere', 'torus', 'mobius'")

    def project_to_sphere(self, x):
        return tf.math.l2_normalize(x, axis=1)

    def project_to_torus(self, x):
        x = tf.math.floormod(x, 2 * np.pi)
        theta = x[:, 0]
        phi = x[:, 1]
        r = 1.0
        R = 2.0
        xt = (R + r * tf.cos(phi)) * tf.cos(theta)
        yt = (R + r * tf.cos(phi)) * tf.sin(theta)
        zt = r * tf.sin(phi)
        return tf.stack([xt, yt, zt], axis=1)

    def project_to_mobius(self, x):
        theta = x[:, 0]
        w = x[:, 1] - 0.5
        xm = (1 + w * tf.cos(theta / 2)) * tf.cos(theta)
        ym = (1 + w * tf.cos(theta / 2)) * tf.sin(theta)
        zm = w * tf.sin(theta / 2)
        return tf.stack([xm, ym, zm], axis=1)
