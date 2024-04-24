import os
import numpy as np
import keras
from keras.models import load_model

image_size = (300, 300)
larger_input = image_size + (3,)
batch_size = 128
num_classes = 8

labels = ["0-3", "4-7", "8-14", "15-20", "21-32", "33-43", "44-53", "54-100"] 


def number_to_string(num):
    if num == 7:
        return "54-100"
    elif num == 6:
        return "44-53"
    elif num == 5:
        return "33-43"
    elif num == 4:
        return "21-32"
    elif num == 3:
        return "15-20"
    elif num == 2:
        return "8-14"
    elif num == 1:
        return "4-7"
    else:
        return "0-3"



if os.path.exists("model5Categorical.keras"):
    # Load the weights if the file exists
    model = load_model("model5Categorical.keras")
    print("Previous weights found, using those")
else:
    print("no model found")


model.summary()

# Initialize a list or NumPy array to store the scores for each class
overall_predictions = [0] * num_classes  # For a list
# class_scores = np.zeros(num_classes)  # For a NumPy array

#folder_path = "larger_categories/8"
#folder_path = "larger_categories_unseen/8"
folder_path = r"C:\Users\ulyss\OneDrive\Desktop\dataset\test"
total_images = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)
        img = keras.utils.load_img(img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        
        
        classes = np.argmax(predictions, axis = 1)
        #print(predictions)
        #print("I think this image belongs to: " + number_to_string(classes[0]) + " age range")
        overall_predictions[classes[0]] += 1
        total_images += 1

# Print the average score for each class
if total_images > 0:
    for i in range(num_classes):
        print("I think there are: " + str(overall_predictions[i]) + " out of " + str(total_images) + " (" + str("{:.2f}".format((overall_predictions[i]/total_images)*100)) + ")% " + labels[i] + " year olds in this folder")
        
else:
    print("No images found in the folder.")

