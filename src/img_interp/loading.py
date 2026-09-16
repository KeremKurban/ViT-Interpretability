"""Model and image loading for the images notebooks.

Thin wrappers over what the original TensorFlow tutorial did inline, so the
notebooks stay about interpretability rather than about data plumbing.
"""
from __future__ import annotations

import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras

INCEPTION_V1 = "https://tfhub.dev/google/imagenet/inception_v1/classification/4"
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

EXAMPLE_IMAGES = {
    "Fireboat": "http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg",
    "Giant Panda": "http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg",
}
# ImageNet class indices used by the tutorial for the two example images
EXAMPLE_TARGETS = {"Fireboat": 555, "Giant Panda": 389}


def load_model():
    """Inception V1 from TF-Hub, wrapped as a Keras model over ``(224, 224, 3)``."""
    inputs = tf_keras.Input(shape=(224, 224, 3))
    outputs = hub.KerasLayer(name="inception_v1", handle=INCEPTION_V1, trainable=False)(inputs)
    return tf_keras.Model(inputs=inputs, outputs=outputs)


def load_imagenet_labels() -> np.ndarray:
    labels_file = tf_keras.utils.get_file("ImageNetLabels.txt", LABELS_URL)
    with open(labels_file) as reader:
        return np.array(reader.read().splitlines())


def read_image(file_name) -> tf.Tensor:
    """Decode, convert to float in [0, 1], and pad-resize to 224x224."""
    image = tf.io.read_file(str(file_name))
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize_with_pad(image, target_height=224, target_width=224)


def load_example_images() -> dict[str, tf.Tensor]:
    paths = {n: tf_keras.utils.get_file(n, url) for n, url in EXAMPLE_IMAGES.items()}
    return {n: read_image(p) for n, p in paths.items()}


def load_baseline_pool(n_images: int = 40, seed: int = 0) -> tf.Tensor:
    """A pool of real photographs to draw Expected Gradients baselines from.

    Uses TensorFlow's ``flower_photos`` set — one reliable download of real
    natural images. They are not ImageNet validation images, which is fine and
    arguably preferable: the baselines should be plausible photographs, not
    examples of the classes being explained. Drawing baselines from the same
    class as the input would leak the answer into the reference point.

    Returns an ``(n_images, 224, 224, 3)`` float tensor in [0, 1].
    """
    archive = tf_keras.utils.get_file("flower_photos.tgz", FLOWERS_URL, extract=True)
    root = pathlib.Path(archive).parent
    candidates = sorted(root.glob("**/*.jpg"))
    if not candidates:
        raise RuntimeError(f"no jpgs found under {root}")

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(candidates), size=min(n_images, len(candidates)), replace=False)
    return tf.stack([read_image(candidates[int(i)]) for i in picks], axis=0)


def top_k_predictions(model, image: tf.Tensor, labels: np.ndarray, k: int = 3):
    probs = tf.nn.softmax(model(tf.expand_dims(image, 0)), axis=-1)
    top_probs, top_idxs = tf.math.top_k(probs, k=k)
    return labels[tuple(top_idxs)], top_probs[0]
