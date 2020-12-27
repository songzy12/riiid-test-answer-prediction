import numpy as np
import tensorflow as tf

# [1,2,3,4] --- w = 2 --[[1,2], [2,3], [3,4]] but 2D to 3D


def rolling_window(a, w):
    s0, s1 = a.strides
    m, n = a.shape
    return np.lib.stride_tricks.as_strided(
        a, shape=(m - w + 1, w, n), strides=(s0, s0, s1)
    )


def make_time_series(x, windows_size):
    x = np.pad(x, [[windows_size - 1, 0], [0, 0]], constant_values=0)
    x = rolling_window(x, windows_size)
    return x


def add_features_to_user(user):
    # We add one to the column in order to have zeros as padding values
    # Start Of Sentence (SOS) token will be 3.
    user["answered_correctly"] = user["answered_correctly"].shift(fill_value=2) + 1
    return user


class RiidSequence(tf.keras.utils.Sequence):
    def __init__(self, users, windows_size, batch_size=256, start=0, end=None):
        self.users = users  # {'user_id': user_df, ...}
        self.windows_size = windows_size
        # to convert indices to our keys
        self.mapper = dict(zip(range(len(users)), users.keys()))
        # start and end to easy generate training and validation
        self.start = start
        self.end = end if end else len(users)
        # To know where the answered_correctly_column is
        self.answered_correctly_index = list(self.user_example().columns).index(
            "answered_correctly"
        )

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        uid = self.mapper[idx + self.start]
        user = self.users[uid].copy()
        y = user["answered_correctly"].to_numpy().copy()
        x = add_features_to_user(user)
        return make_time_series(x, self.windows_size), y

    def user_example(self):
        """Just to check what we have till' now."""
        uid = self.mapper[self.start]
        return add_features_to_user(self.users[uid].copy())

    # INFERENCE PART
    def get_user_for_inference(self, user_row):
        """Picks a new user row and concats it to previous interactions
        if it was already stored.

        Maybe the biggest trick in the notebook is here. We reuse the user_id column to
        insert the answered_correctly SOS token because we previously placed the column
        there on purpose.

        After it, we roll that column and then crop it if it was bigger than the window
        size, making the SOS token disapear if out of the sequence.

        If the sequence if shorter than the window size, then we pad it.
        """
        uid = user_row[self.answered_correctly_index]
        user_row[self.answered_correctly_index] = 2  # SOS token
        user_row = user_row[np.newaxis, ...]
        if uid in self.users:
            x = np.concatenate([self.users[uid], user_row])
            # same as in training, we need to add one!!!
            x[:, self.answered_correctly_index] = (
                np.roll(x[:, self.answered_correctly_index], 1) + 1
            )
        else:
            x = user_row

        if x.shape[0] < self.windows_size:
            return np.pad(x, [[self.windows_size - x.shape[0], 0], [0, 0]])
        elif x.shape[0] > self.windows_size:
            return x[-self.windows_size :]
        else:
            return x

    def update_user(self, uid, user):
        """Concat the new user's interactions to the old ones if already stored."""
        if uid in self.users:
            self.users[uid] = np.concatenate([self.users[uid], user])[
                -self.windows_size :
            ]
        else:
            self.users[uid] = user


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# NN THINGS


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -1e9
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(dff, activation="relu"), tf.keras.layers.Dense(d_model)]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


def create_padding_mask(seqs):
    # We mask only those vectors of the sequence in which we have all zeroes
    # (this is more scalable for some situations).
    mask = tf.cast(tf.reduce_all(tf.math.equal(seqs, 0), axis=-1), tf.float32)
    # (batch_size, 1, 1, seq_len)
    return mask[:, tf.newaxis, tf.newaxis, :]


def get_series_model(
    n_features,
    content_ids,
    task_container_ids,
    part_ids,
    windows_size=64,
    d_model=24,
    num_heads=4,
    n_encoder_layers=2,
):
    # Input
    inputs = tf.keras.Input(shape=(windows_size, n_features), name="inputs")
    mask = create_padding_mask(inputs)
    pos_enc = positional_encoding(windows_size, d_model)

    # Divide branches
    content_id = inputs[..., 0]
    task_container_id = inputs[..., 1]
    answered_correctly = inputs[..., 2]
    elapsed_time = inputs[..., 3]
    part = inputs[..., 4]

    # Create embeddings
    content_embeddings = tf.keras.layers.Embedding(content_ids, d_model)(content_id)
    task_embeddings = tf.keras.layers.Embedding(task_container_ids, d_model)(
        task_container_id
    )
    answered_correctly_embeddings = tf.keras.layers.Embedding(4, d_model)(
        answered_correctly
    )
    # Continuous! Only a learnable layer for it.
    elapsed_time_embeddings = tf.keras.layers.Dense(d_model, use_bias=False)(
        elapsed_time
    )
    part_embeddings = tf.keras.layers.Embedding(part_ids, d_model)(part)

    # Add embeddings
    x = tf.keras.layers.Add()(
        [
            pos_enc,
            content_embeddings,
            task_embeddings,
            answered_correctly_embeddings,
            elapsed_time_embeddings,
            part_embeddings,
        ]
    )

    for _ in range(n_encoder_layers):
        x = EncoderLayer(
            d_model=d_model, num_heads=num_heads, dff=d_model * 4, rate=0.1
        )(x, mask=mask)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, output, name="model")
