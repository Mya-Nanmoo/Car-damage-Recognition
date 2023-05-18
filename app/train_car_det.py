import tensorflow as tf
import numpy as np
import matplotlib as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Load the CIFAR-10 dataset
def load_cifar10_data():
    """
    Load the CIFAR-10 dataset from the Keras library and normalize the pixel values.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    selected_classes = [0, 1, 8, 9]

    train_mask = np.isin(y_train, selected_classes).flatten()
    test_mask = np.isin(y_test, selected_classes).flatten()
    
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cifar10_data()

# Create the CNN model
model = tf.keras.Sequential(name='CAR_DETECTOR')
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name="ConvLayer_01"))
model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), name='PoolingLayer_01'))
model.add(tf.keras.layers.BatchNormalization(axis = 3, name='Bn_conv1'))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',name='ConvLayer_02'))
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=(1,1), name='PoolingLayer_02'))
model.add(tf.keras.layers.BatchNormalization(axis = 3, name='Bn_conv02'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu',name='DenseLayer_01'))
model.add((tf.keras.layers.Dropout(0.50)))

model.add(tf.keras.layers.Dense(64, activation='relu',name='DenseLayer_02'))
model.add((tf.keras.layers.Dropout(0.30)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

epochs=20
adam = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
rmsprop=tf.keras.optimizers.RMSprop(learning_rate=0.0001)
sparse_categorical_cross_entropy=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy', metrics=["accuracy"])

# using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath = "resources/checkpoints/Object_Detector.hdf5", verbose = 1, save_best_only = True)

history = model.fit(x_train, y_train, batch_size=5, epochs=epochs, validation_split=0.30, callbacks=[checkpointer, earlystopping])

print("\n***DONE***")

# Plot the training and validation accuracy over time
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('static/car_accuracy_progression.jpg', dpi=300, quality=80, optimize=True, progressive=True) 

# Plot the training and validation loss over time
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('static/car_loss_progression.jpg', dpi=300, quality=80, optimize=True, progressive=True) 

model.save("resources/models/Object_Detector.h5")