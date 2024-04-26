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
        output_activation = 'sigmoid'
        units = 1
    else:
        output_activation = 'softmax'
        units = num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=output_activation)(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Model loading or creation
modelName = "MaskIdentificationModel"
model_file_path = f"{modelName}.keras"
if os.path.exists(model_file_path):
    model = load_model(model_file_path)
    print("Previous weights found, using those")
else:
    model = create_model(input_shape=image_size + (3,), num_classes=2)
    print("No previous weights found, starting fresh")

# Model summary
model.summary()

# Train the model
epochs = 6
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, shuffle=True)

# Save the trained model
model.save(model_file_path)

# Plot training and validation accuracy and losses
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Validation')
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(loc='upper left')

ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Validation')
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(loc='upper left')

plt.savefig("plot.png")

plt.show()

# Evaluate the model on training and validation sets
train_loss, train_acc = model.evaluate(train_ds)
val_loss, val_acc = model.evaluate(val_ds)
print(f"Final train accuracy = {train_acc * 100:.2f}%, Validation accuracy = {val_acc * 100:.2f}%")