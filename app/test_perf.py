import tensorflow as tf

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.models.load_model("resources/models/Object_Detector.h5")

def model_performance(model):
    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return test_loss, test_acc
