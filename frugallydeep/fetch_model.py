import tensorflow as tf

model = tf.keras.models.load_model("tf_cifar10.h5")

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
