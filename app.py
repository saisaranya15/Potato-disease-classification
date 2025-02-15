from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# Paths
MODEL_PATH = "potato_disease_model.h5"
DATASET_PATH = "PlantVillage"  # Updated dataset path
UPLOAD_FOLDER = "static"  # For storing uploaded images
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CLASS_LABELS = ["early blight", "healthy", "late blight"]

# Ensure static folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Train model if not found
if not os.path.exists(MODEL_PATH):
    print("Training model as no saved model found...")

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_ds = train_datagen.flow_from_directory(
        DATASET_PATH, target_size=(224, 224), batch_size=32, class_mode="categorical", subset="training"
    )
    val_ds = train_datagen.flow_from_directory(
        DATASET_PATH, target_size=(224, 224), batch_size=32, class_mode="categorical", subset="validation"
    )

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1)

    # Save model
    model.save(MODEL_PATH)
    print("Model trained and saved successfully!")

else:
    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded saved model.")

# Function to predict image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return CLASS_LABELS[predicted_class]

# Homepage route
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("index.html", filename=file.filename, prediction=prediction)

    return render_template("index.html", filename=None, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
