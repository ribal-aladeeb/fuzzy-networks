import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

# I'm aware that the train/test split is not the same as in the previous file
# but the purpose is simply to track dependencies during inference.
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

model = tf.keras.models.load_model('fashion_mnist_model.h5')
prediction = model.predict(test_images[:1])
print(f'model prediction {prediction}')

probability_model = tf.keras.Sequential(
    [model, tf.keras.layers.Softmax()]
    )

prediction = probability_model.predict(test_images[:1])
print(f'probability prediction {prediction}')