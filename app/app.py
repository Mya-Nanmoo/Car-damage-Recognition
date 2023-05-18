from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('resources/models/Object_Detector.h5')
model2 = load_model('resources/models/Damage_Detector.h5')

# Create the Flask app
app = Flask(__name__)

# Define the endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Open the image file and convert to RGB format
    image = Image.open(file).convert('RGB')

    # Resize the image to match the input shape of the model
    image_ph1 = image.resize((32, 32))

    # Convert the image to a numpy array and normalize the pixel values
    x = np.array(image_ph1) / 255.0

    # Make a prediction using the trained model
    y = model.predict(np.array([x]))

    # Get the predicted class label
    label = np.argmax(y[0])
    if label == 1:
        image_ph2 = image.resize((64, 64))
        x_ = np.array(image_ph2) / 255.0
        y_ = model2.predict(np.array([x_]))
        # Get the predicted class label
        label = np.argmax(y_[0])
        label_dict = {0: "Damaged",1: "Clean"}
        # Return the predicted class label as a JSON response
        return jsonify({'res': label_dict[label]})
    else:
        return jsonify({'res': "Not a Car!!"})

# Render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run()

