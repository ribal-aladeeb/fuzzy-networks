import tensorflow as tf
from tensorflow.keras import datasets, layers, models


(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.applications.MobileNetV2(
    input_shape=train_images.shape[1:], include_top=True, alpha=1.0, weights=None,
    input_tensor=None, pooling=None,  classifier_activation='softmax',
)


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

num_epochs = 100
history = model.fit(train_images, train_labels, epochs=num_epochs,
                    validation_data=(test_images, test_labels),batch_size=128)

model.save(f'mobilenetv2_{num_epochs}_epochs.h5')
