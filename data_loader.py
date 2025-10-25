import tensorflow as tf
from tensorflow.keras import layers
from config import config

def load_datasets():

    full_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,

        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    class_names = full_ds.class_names
    print("\nClasses found:", class_names)

    # Split into train+val (80%) and test (20%)
    train_val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )

    # Split train_val into train (80%) and val (20%)
    train_size = int(0.8 * len(list(train_val_ds)))
    train_ds = train_val_ds.take(train_size)
    val_ds = train_val_ds.skip(train_size)

    # Cache/prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def preprocess_datasets(train_ds, val_ds, test_ds):
    """Normalize and flatten datasets"""
    norm = layers.Rescaling(1. / 255)

    def map_norm_flat(ds):
        return ds.map(lambda x, y: (tf.reshape(norm(x), [tf.shape(x)[0], -1]), y))

    train_ds_flat = map_norm_flat(train_ds)
    val_ds_flat = map_norm_flat(val_ds)
    test_ds_flat = map_norm_flat(test_ds)

    INPUT_DIM = config.IMG_SIZE[0] * config.IMG_SIZE[1] * 3

    return train_ds_flat, val_ds_flat, test_ds_flat, INPUT_DIM