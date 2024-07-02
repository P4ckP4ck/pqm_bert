import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from network import create_bert_model, masked_mse_loss, create_masked_input
from helper import create_sgdr_scheduler, cosine_decay_with_warm_restart, prepare_data
import config as cfg


if __name__ == "__main__":
    data_raw = np.load("training_data/x_train.npy")
    data = prepare_data(data_raw)

    model = create_bert_model(cfg.input_shape)
    optimizer = Adam(learning_rate=cfg.initial_learning_rate)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    try:
        model.load_weights("bert_pre_train.h5")
    except FileNotFoundError:
        print("No weights found. Initializing new weights!")

    sgdr_scheduler = create_sgdr_scheduler(cfg.initial_learning_rate, cfg.cycle_length, cfg.learning_rate_min)
    global_step = 0

    for epoch in range(cfg.epochs):
        new_lr = cosine_decay_with_warm_restart(cfg.initial_learning_rate, epoch, cfg.cycle_length, cfg.learning_rate_min)
        tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
        print(f"Epoch {epoch + 1}/{cfg.epochs} --- Learning Rate: {new_lr:.2e}")

        for i in range(0, len(data), cfg.batch_size):
            batch_data = data[i:i + cfg.batch_size]
            masked_data, labels = create_masked_input(batch_data)
            loss = model.train_on_batch(masked_data, labels)
            print(f"Batch {i // cfg.batch_size + 1}/{len(data) // cfg.batch_size}, Loss: {loss}")
            global_step += 1
        if epoch % 10 == 0:
            model.save_weights("bert_pre_train.h5")
