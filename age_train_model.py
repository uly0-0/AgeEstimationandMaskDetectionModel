

"""
## Setup
"""

import os
import numpy as np
import keras
from keras import layers
from keras.models import load_model

"""
## Generate a `Dataset`
"""

image_size = (300, 300)
larger_input = image_size + (3,)
batch_size = 32


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # we don't expect Bo to be upside-down so we will not flip vertically
        


# load and iterate training dataset
train_ds = datagen.flow_from_directory(r"C:\Users\ulyss\OneDrive\Desktop\dataset\train", 
                                       target_size=image_size, 
                                       color_mode='rgb', 
                                       class_mode='categorical', 
                                       batch_size=batch_size)
# load and iterate validation dataset
val_ds = datagen.flow_from_directory(r"C:\Users\ulyss\OneDrive\Desktop\dataset\validation", 
                                      target_size=image_size, 
                                      color_mode='rgb', 
                                      class_mode='categorical', 
                                      batch_size=batch_size)


modelName = "model5Categorical"
def create_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Display the model's architecture

#keras.utils.plot_model(model, show_shapes=True)

if os.path.exists(f"{modelName}.keras"):
    # Load the weights if the file exists
    model = load_model(f"{modelName}.keras")
    print("Previous weights found, using those")
else:
    model = create_model(input_shape=image_size + (3,), num_classes=8)
    print("no previous weights, starting fresh")



model.summary()

"""
## Train the model
"""

epochs = 1
loops = 15


for i in range(loops):
    model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    shuffle=True,
    )

    model.save(f"{modelName}.keras")
