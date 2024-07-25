import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
from albumentations import (Compose, RandomCrop, HorizontalFlip, RandomBrightnessContrast, GridDistortion, OpticalDistortion)
from lime import lime_image
from skimage.segmentation import mark_boundaries
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define paths
dataset_dir = 'C:/Brain MRI Project/dataset'  # Update this to your dataset path

# Create directories for training and testing
train_dir = os.path.join(dataset_dir, 'Training')
test_dir = os.path.join(dataset_dir, 'Testing')

# Data Preprocessing
img_size = (224, 224)  # Resize images to 224x224

# Advanced Data Augmentation using Albumentations
def advanced_augmentations(image):
    augmentations = Compose([
        RandomCrop(width=224, height=224),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        GridDistortion(p=0.2),
        OpticalDistortion(p=0.2)
    ])
    return augmentations(image=image)['image']

datagen_train = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=advanced_augmentations
)

datagen_test = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train.flow_from_directory(
    directory=train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

test_generator = datagen_test.flow_from_directory(
    directory=test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

# Model Building
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
x = Dense(512, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze some layers in the base model for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Training
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Save the model
model.save('final_model.h5')

# Visualization of Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

# Prediction Function
def predict_image(model, img_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    class_name = class_labels[class_idx]
    confidence = predictions[0][class_idx]
    
    return class_name, confidence

# Function to visualize image and prediction
def visualize_prediction(img_path, class_name, confidence):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'Predicted: {class_name} ({confidence:.2f})')
    plt.axis('off')
    plt.show()

# LIME Explanation Function
def explain_prediction_lime(model, img_path, num_samples=1000):
    """
    Generate and visualize LIME explanation for the given image.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    def model_predict(images):
        preds = model.predict(images)
        return preds

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array[0], model_predict, top_labels=1, hide_color=0, num_samples=num_samples)

    temp, mask = explanation.get_image_and_mask(label=0, positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp, mask)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_boundry)
    plt.title('LIME Explanation')
    plt.axis('off')
    plt.show()

# Example usage: Make prediction, visualize result, and LIME explanation
img_path = 'C:/Brain MRI Project/dataset/Testing/glioma/Te-gl_0013.jpg'  # Update this path to your image
class_name, confidence = predict_image(model, img_path)
visualize_prediction(img_path, class_name, confidence)

# LIME Explanation
explain_prediction_lime(model, img_path)
