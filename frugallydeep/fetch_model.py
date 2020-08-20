import tensorflow as tf
import sys

try:
    model_filename = sys.argv[1]
except Exception as e:
    print(e)
    print("please provide a model filename to calculate test set accuracy")
    exit()

model = tf.keras.models.load_model(model_filename)


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
