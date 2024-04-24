import cv2
import argparse
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os
# load pre-trained super resolution
#model = tf.keras.applications.VGG19(weights = 'imagenet', include_top = False)

"""def upscale_image(image_path):
    # load & process image
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (224, 224))
    input_image = np.expand_dims(input_image, axis = 0)

    #upscale image
    upscaled_image = model.predict(input_image)

    # post process the upscaled image
    upscaled_image = np.clip(upscaled_image[0], 0, 225).astype(np.uint8)

    return upscaled_image

def save_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Simple Image Upscaler")
    parser.add_argument("input", help = "Path to the input image file")
    parser.add_argument("output", help="Path to save upscaled image file")

    args = parser.parse_args()
    
    input_image_path = args.input
    output_image_path = args.output # replace with desired output path

    upscale_image(input_image_path, output_image_path)
"""

#testing commit omar
#testing commit uly
# Load dataset metadata
metadata = pd.read_csv(r'C:\Users\ulyss\OneDrive\Documents\FresnoState\Spring 2024\CSCI 158- Biometric Security\FaceRecognition-AgeEstimation') # Update with actual path to your meta.csv file

# Assuming 'path' column points to image files, 'age' is the age, and 'gender' is the gender (0 for female, 1 for male)
dataset_directory = r'C:\Users\ulyss\OneDrive\Documents\FresnoState\Spring 2024\CSCI 158- Biometric Security\dataset' # Update with the actual directory of your images

# Splitting dataset into training and validation
train_df = metadata.sample(frac=0.8, random_state=200)  # Adjust frac as needed
validation_df = metadata.drop(train_df.index)

#modify 'path' column to include the subfolder information
train_df['path'] = train_df['path'].apply(lambda x: os.path.join(dataset_directory, x))
validation_df ['path'] = validation_df['path'].apply(lambda x: os.path.join(dataset_directory, x))

# Data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
#fix column issue
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory= dataset_directory,
    x_col="path",
    y_col=["age", "gender"],
    target_size=(128, 128),
    batch_size=32,
    class_mode="multi_output")

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=dataset_directory,
    x_col="path",
    y_col=["age", "gender"],
    target_size=(128, 128),
    batch_size=32,
    class_mode="multi_output")

def build_model():
    input_img = Input(shape=(128, 128, 3))
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layers
    age_output = Dense(1, name='age_output')(x)  # No activation, as this is a regression problem
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)  # Binary classification
    
    model = Model(inputs=input_img, outputs=[age_output, gender_output])
    
    return model

model = build_model()
model.compile(optimizer='adam',
              loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Adjust based on your dataset size
    epochs=10,  # Increase epochs for better results
    validation_data=validation_generator,
    validation_steps=50)  # Adjust based on your dataset size
