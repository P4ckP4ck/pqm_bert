import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model



class PositionalEncoding(layers.Layer):
    def __init__(self, maxlen, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding.numpy(),  # Convert tensor to numpy array for serialization
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def transformer_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attention_output = layers.Dropout(dropout)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

    ffn = layers.Dense(ff_dim, activation='gelu')(attention_output)
    ffn_output = layers.Dense(d_model)(ffn)
    ffn_output = layers.Dropout(dropout)(ffn_output)

    return layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)


def create_bert_model(input_shape, d_model=128, num_heads=2, ff_dim=128, num_transformer_blocks=3):
    input_layer = layers.Input(shape=input_shape)
    embedding_layer = layers.Dense(d_model, activation='relu')(input_layer)

    maxlen = input_shape[0]
    positional_encoding_layer = PositionalEncoding(maxlen, d_model)
    x = positional_encoding_layer(embedding_layer)

    for _ in range(num_transformer_blocks):
        x = transformer_block(x, d_model, num_heads, ff_dim)

    output_layer = layers.Dense(input_shape[-1])(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def create_masked_input(input_data, mask_probability=0.15):
    mask = np.random.rand(*input_data.shape) < mask_probability
    masked_data = input_data.copy()
    masked_data[mask] = 0
    labels = np.where(mask, input_data, -1)
    return masked_data, labels


def masked_mse_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    loss = tf.keras.losses.mean_squared_error(masked_y_true, masked_y_pred)
    return loss
