import tensorflow as tf
import os
import matplotlib as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Define paths to the data folders
train_path = '/home/brian/Downloads/Object Detector/training'# directory for your train data.
val_path = '/home/brian/Downloads/Object Detector/validation' # directory for your validation data.

# Define a function to create a dataset from a directory path
def create_dataset_from_dir(path):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        batch_size=32,
        image_size=(256, 256),
        validation_split=0.2,
        subset='training',
        seed=123
    )
    return dataset

# Create training and validation datasets
train_dataset = create_dataset_from_dir(train_path)
val_dataset = create_dataset_from_dir(val_path)

# Define a function to preprocess the images
def preprocess(image, label):
    image = tf.image.resize(image, (64, 64)) # Resize the image
    image = tf.cast(image, tf.float32) # Convert the image to a float32
    image = image / 255.0 # Normalize the pixel values to be between 0 and 1
    return image, label

# Preprocess the images in the datasets
train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# Define the model architecture
# Create the CNN model
model2 = tf.keras.Sequential(name='DAMAGE_DETECTOR')
model2.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3), name="ConvLayer_01"))
model2.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), name='PoolingLayer_01'))
model2.add(tf.keras.layers.BatchNormalization(axis = 3, name='Bn_conv1'))

model2.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu',name='ConvLayer_02'))
model2.add(tf.keras.layers.MaxPooling2D((2,2), strides=(1,1), name='PoolingLayer_02'))
model2.add(tf.keras.layers.BatchNormalization(axis = 3, name='Bn_conv02'))

model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(128, activation='relu',name='DenseLayer_01'))
model2.add((tf.keras.layers.Dropout(0.50)))

model2.add(tf.keras.layers.Dense(64, activation='relu',name='DenseLayer_02'))
model2.add((tf.keras.layers.Dropout(0.30)))

model2.add(tf.keras.layers.Dense(16, activation='relu',name='DenseLayer_03'))

model2.add(tf.keras.layers.Dense(2, activation='sigmoid'))

epochs=20

adam = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
#rmsprop=tf.keras.optimizers.RMSprop(learning_rate=0.0001)

sparse_categorical_cross_entropy=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model2.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath = "resources/checkpoints/Damage_Detector.hdf5", verbose = 1, save_best_only = True)

history = model2.fit(train_dataset, batch_size=32, epochs=epochs, validation_data=val_dataset, callbacks=[checkpointer, earlystopping])
print("\n***DONE***")

# Plot the training and validation accuracy over time
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('static/damage_accuracy_progression.jpg', dpi=300, quality=80, optimize=True, progressive=True) 

# Plot the training and validation loss over time
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('static/damage_loss_progression.jpg', dpi=300, quality=80, optimize=True, progressive=True) 

model2.save("resources/models/Damage_Detector.h5")