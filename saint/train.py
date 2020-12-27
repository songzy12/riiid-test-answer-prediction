from data_processor import get_df
from model import *

path_questions = "../input/riiid-test-answer-prediction/questions.csv"
path_train = "../input/riiid-test-answer-prediction/train.csv"
df, part_ids, content_ids, task_container_ids = get_df(path_questions, path_train)

train_idx = int(len(df) * 0.8)
windows_size = 64
epochs = 300
patience = 2
d_model = 32
num_heads = 4
n_encoder_layers = 2

s_train = RiidSequence(df, windows_size, start=0, end=train_idx)
s_val = RiidSequence(df, windows_size, start=train_idx)

n_features = s_train[0][0].shape[-1]

tf.keras.backend.clear_session()
model = get_series_model(
    n_features,
    content_ids,
    task_container_ids,
    part_ids,
    windows_size=windows_size,
    d_model=d_model,
    num_heads=num_heads,
    n_encoder_layers=n_encoder_layers,
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="AUC"),
        tf.keras.metrics.BinaryAccuracy(name="acc"),
    ],
)

# tf.keras.utils.plot_model(model)

model.fit(
    s_train,
    validation_data=s_val,
    epochs=epochs,
    workers=4,
    shuffle=True,
    use_multiprocessing=True,
    callbacks=tf.keras.callbacks.EarlyStopping(
        patience=patience, monitor="val_AUC", mode="max", restore_best_weights=True
    ),
    verbose=1,
)
model.save_weights("model.h5")
