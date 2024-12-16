# Import Required Libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Update these paths based on your local machine
image_dir = 'C:/Users/IC404/PycharmProjects/FashionAi/images'
shape_labels = 'C:/Users/IC404/PycharmProjects/FashionAi/labels/shape/shape_anno_all.txt'
fabric_labels = 'C:/Users/IC404/PycharmProjects/FashionAi/labels/texture/fabric_ann.txt'
pattern_labels = 'C:/Users/IC404/PycharmProjects/FashionAi/labels/texture/pattern_ann.txt'

# Load Shape Annotations
print("Loading shape annotations...")
shape_df = pd.read_csv(shape_labels, sep=' ', header=None, names=['image', 'shape_0', 'shape_1', 'shape_2', 'shape_3', 'shape_4',
                                                                  'shape_5', 'shape_6', 'shape_7', 'shape_8', 'shape_9',
                                                                  'shape_10', 'shape_11'])
print(f"Shape annotations loaded with shape: {shape_df.shape}")

# Load Fabric Annotations
print("Loading fabric annotations...")
fabric_df = pd.read_csv(fabric_labels, sep=' ', header=None, names=['image', 'upper_fabric', 'lower_fabric', 'outer_fabric'])
print(f"Fabric annotations loaded with shape: {fabric_df.shape}")

# Load Pattern Annotations
print("Loading pattern annotations...")
pattern_df = pd.read_csv(pattern_labels, sep=' ', header=None, names=['image', 'upper_color', 'lower_color', 'outer_color'])
print(f"Pattern annotations loaded with shape: {pattern_df.shape}")

# Merge all Annotations
print("Merging all annotations...")
labels = shape_df.merge(fabric_df, on='image').merge(pattern_df, on='image')
print(f"Merged annotations shape: {labels.shape}")

# Create a cloth type category
print("Inferring cloth types...")
def infer_cloth_type(row):
    if row['shape_0'] == 1 and row['upper_fabric'] == 1:
        return 't-shirt'
    elif row['shape_0'] == 3 and row['upper_fabric'] == 0:
        return 'jacket'
    else:
        return 'unknown'

labels['cloth_type'] = labels.apply(infer_cloth_type, axis=1)
labels['cloth_type_id'] = labels['cloth_type'].astype('category').cat.codes
print(f"Cloth types inferred. Unique types: {labels['cloth_type'].unique()}")

# --- STEP 2: SPLIT DATASET ---
print("Splitting dataset into training and validation...")
# Stratified split into train and validation
train_data, val_data = train_test_split(labels, test_size=0.2, stratify=labels[['cloth_type_id']])

print(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

# --- STEP 3: IMAGE LOADING FUNCTION ---
def load_images(data, img_dir, img_size=(224, 224)):
    images = []
    cloth_targets = []
    print("Loading images...")
    for _, row in data.iterrows():
        img_path = os.path.join(img_dir, row['image'].replace('\\', '/'))  # Normalize path separators
        print(f"Checking path: {img_path}")  # DEBUG: Check the constructed path
        if not os.path.isfile(img_path):
            print(f"Image not found: {img_path}")
            continue  # Skip missing files
        try:
            img = load_img(img_path, target_size=img_size)  # Resize image
            img_array = img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            cloth_targets.append(row['cloth_type_id'])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    print(f"Loaded {len(images)} images from {len(data)} entries.")
    return np.array(images), np.array(cloth_targets)

# Load training and validation images
x_train, y_train_cloth = load_images(train_data, image_dir)
x_val, y_val_cloth = load_images(val_data, image_dir)

print(f"Training images shape: {x_train.shape}, Validation images shape: {x_val.shape}")

# --- STEP 4: BUILD CLOTH TYPE CLASSIFICATION MODEL ---
def build_cloth_type_model(input_shape=(224, 224, 3), num_cloth_types=5):
    print("Building cloth type classification model...")
    inputs = layers.Input(shape=input_shape)

    # Shared CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Output for cloth type classification
    cloth_type_output = layers.Dense(num_cloth_types, activation='softmax', name='cloth_type_output')(x)

    # Model
    model = models.Model(inputs=inputs, outputs=cloth_type_output)
    print("Model built successfully.")
    return model

model = build_cloth_type_model(num_cloth_types=len(labels['cloth_type_id'].unique()))

model.summary()

# --- STEP 5: COMPILE AND TRAIN ---
print("Compiling the model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled successfully.")

# Train the model
if x_train.size > 0:
    print("Starting training...")
    history = model.fit(
        x_train,
        y_train_cloth,
        validation_data=(x_val, y_val_cloth),
        epochs=20,
        batch_size=32
    )
    print("Training completed.")

    # --- STEP 6: SAVE AND EVALUATE ---
    # Save the model
    print("Saving the model...")
    model.save('./cloth_type_model.h5')
    print("Model saved successfully!")

    # Evaluate the model
    print("Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(x_val, y_val_cloth)
    print(f"Validation Loss: {test_loss}, Validation Accuracy: {test_accuracy}")

    # --- STEP 7: PLOT TRAINING RESULTS ---
    def plot_training_history(history):
        print("Plotting training history...")
        plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title('Cloth Type Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    plot_training_history(history)
else:
    print("No training data available. Check your dataset paths and annotations.")
