import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from PIL import Image

# Load the Titanic dataset (train and test)

train_data = pd.read_csv(r'C:\Users\Darshan\OneDrive\Desktop\Program\Titanic ML prediction\titanic\train.csv')
test_data = pd.read_csv(r'C:\Users\Darshan\OneDrive\Desktop\Program\Titanic ML prediction\titanic\test.csv')

# Select the relevant features and target variable (Survived)

target = train_data["Survived"]
selected_features = ["Pclass", "Sex", "SibSp", "Parch"]

# Convert categorical features like 'Sex' to numeric using one-hot encoding

X_train = pd.get_dummies(train_data[selected_features])
X_test = pd.get_dummies(test_data[selected_features])

# Train a Random Forest model to predict survival

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf_model.fit(X_train, target)

# Create a directory to save the individual tree images

tree_images_folder = "decision_trees"
if not os.path.exists(tree_images_folder):
    os.makedirs(tree_images_folder)

# Loop through all trees in the random forest and save each one as a PNG image

for i, tree in enumerate(rf_model.estimators_):
    # Set up the plot for the current tree

    plt.figure(figsize=(20, 10))
    
    # Plot the tree with the feature names and class labels

    plot_tree(tree, feature_names=X_train.columns, class_names=["Did not survive", "Survived"], filled=True, rounded=True)
    
    # Save the current tree plot as a PNG file

    tree_filename = os.path.join(tree_images_folder, f'decision_tree_{i + 1}.png')
    plt.savefig(tree_filename)
    plt.close()  # Close the plot to free memory

    print(f"Tree {i + 1} saved as '{tree_filename}'")

# Combine all the individual decision tree images into one large image (grid format)
# Load the saved tree images into a list
tree_images = []
for i in range(len(rf_model.estimators_)):
    img_path = os.path.join(tree_images_folder, f'decision_tree_{i + 1}.png')
    tree_images.append(Image.open(img_path))

# Calculate the grid size (rows and columns) based on the number of trees
num_trees = len(tree_images)
num_columns = 3  # Define how many trees per row
num_rows = (num_trees // num_columns) + (num_trees % num_columns > 0)  # Calculate number of rows needed

# Create a blank canvas for the combined image

tree_widths, tree_heights = zip(*(img.size for img in tree_images))
max_tree_width = max(tree_widths)
max_tree_height = max(tree_heights)

# Create a new image with enough space to fit all the trees

combined_image = Image.new('RGB', (max_tree_width * num_columns, max_tree_height * num_rows))

# Position the trees in the grid

x_offset = 0
y_offset = 0
for i, img in enumerate(tree_images):
    combined_image.paste(img, (x_offset, y_offset))
    
    # Move to the next column, or reset to the first column and move to the next row
    x_offset += max_tree_width
    if (i + 1) % num_columns == 0:
        x_offset = 0
        y_offset += max_tree_height

# Save the combined image
combined_image_filename = "combined_decision_trees.png"
combined_image.save(combined_image_filename)

print(f"All trees combined and saved as '{combined_image_filename}'")

# Optionally, display the combined image
plt.imshow(combined_image)
plt.axis('off')  # Turn off axis
plt.show()
