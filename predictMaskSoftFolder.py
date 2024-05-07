import os
import numpy as np
import keras
from keras.models import load_model

image_size = (300, 300)
larger_input = image_size + (3,)
batch_size = 128
num_classes = 2

if os.path.exists("modelBinary.keras"):
    # Load the weights if the file exists
    model = load_model("modelBinary.keras")
    print("Previous weights found, using those")
else:
    print("no model found")


model.summary()

# Initialize a list or NumPy array to store the scores for each class
overall_predictions = [0] * num_classes  # For a list
# class_scores = np.zeros(num_classes)  # For a NumPy array

#folder_path = "larger_categories/8"
#folder_path = "larger_categories_unseen/8"
folder_path = r"C:\Users\omar2\OneDrive\Desktop\Mask Identification Dataset\Face Mask Dataset\Validation\1"
total_images = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        img_path = os.path.join(folder_path, filename)
        img = keras.utils.load_img(img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        
        score = float(predictions[0][0])
        overall_predictions[0] += (1-score)
        overall_predictions[1] += score
        total_images += 1
        print(f"This image is {100 * (1 - score):.2f}% mask and {100 * score:.2f}% non-mask.")

# Print the average score for each class
if total_images > 0:
    #print("I think there are: " + str(overall_predictions[i]) + " out of " + str(total_images) + " (" + str("{:.2f}".format((overall_predictions[i]/total_images)*100)) + ")% " + labels[i] + " year olds in this folder")
    print(f"This folder is {100 * (overall_predictions[0]/total_images):.2f}% mask and {100 * (overall_predictions[1]/total_images):.2f}% non-mask.")

else:
    print("No images found in the folder.")