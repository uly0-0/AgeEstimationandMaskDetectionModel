import os
import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup
image_size = (300, 300)
batch_size = 32

# Data generators
datagen = ImageDataGenerator(
    samplewise_center=True,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

train_ds = datagen.flow_from_directory(
    r"C:\Users\omar2\OneDrive\Desktop\Mask Identification Dataset\Face Mask Dataset\Train",
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary', 
    batch_size=batch_size
)

val_ds = datagen.flow_from_directory(
    r"C:\Users\omar2\OneDrive\Desktop\Mask Identification Dataset\Face Mask Dataset\Validation",
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary',  
    batch_size=batch_size
)

# Model creation
def create_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x  

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation="sigmoid")(x)  
    model = keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])  
    return model

# Model loading or creation
modelName = "model5Binary"
if os.path.exists(f"{modelName}.keras"):
    model = load_model(f"{modelName}.keras")
    print("Previous weights found, using those")
else:
    model = create_model(input_shape=image_size + (3,), num_classes=2)
    print("No previous weights found, starting fresh")

# Model summary
model.summary()

# Train the model
epochs = 1
loops = 15

for i in range(loops):
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        shuffle=True,
    )

    # Save the trained model
    model.save(f"{modelName}.keras")

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()