import os
import keras
from keras.models import load_model

image_size = (300, 300)
larger_input = image_size + (3,)
batch_size = 128
num_classes = 8

labels = ["0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"] 


def number_to_string(num):
    if num == 7:
        return "60+"
    elif num == 6:
        return "48-53"
    elif num == 5:
        return "38-43"
    elif num == 4:
        return "25-32"
    elif num == 3:
        return "15-20"
    elif num == 2:
        return "8-13"
    elif num == 1:
        return "4-6"
    else:
        return "0-2"



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
folder_path = "professor"
total_images = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)
        img = keras.utils.load_img(img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        for i in range(num_classes):
            overall_predictions[i] += predictions[0][i]
        
        total_images += 1

# Print the average score for each class
if total_images > 0:
    for i in range(num_classes):
        print(str("{:.10f}".format(overall_predictions[i]/total_images)) + " " + labels[i])
        
else:
    print("No images found in the folder.")

