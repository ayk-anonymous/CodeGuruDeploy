from flask import Flask, request, jsonify, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Replace 'path/to/your/h5_model_file.h5' with the actual path to your h5 model file
effnetb0_model = 'model/effnetb0_model_v3.h5'
codebert_model = 'model/codebert_model_v3.h5'

# Load the model using tf.keras.models.load_model
effnetb0loaded_model = tf.keras.models.load_model(effnetb0_model)
codebertloaded_model = tf.keras.models.load_model(codebert_model)

@app.route('/')
def index():
    return render_template('index-image.html')

@app.route('/text')
def text():
    return render_template('index-text.html')

@app.route('/image')
def image():
    return render_template('index-image.html')

@app.route('/predict_image', methods=['POST'])

def predict_image():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        filename = image_file.filename  # Get the filename
        img = Image.open(image_file)

        
        # Resize the image to match the input size expected by the model
        img = img.resize((224, 224))

        # Convert to RGB (discarding alpha channel)
        img = img.convert('RGB')
        
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)

        # Expand dimensions to match the model's input shape
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions using the loaded model
        prediction = effnetb0loaded_model.predict(img_array)

        class_labels = ["O(logn)", "O(n^2)", "O(n^3)", "O(1)", "O(n)", "O(nlogn)", "Unknown", "O(2^n)"]

# Map probabilities to class labels
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
# Render the template with the predicted class label
        return render_template('predict-image.html', class_label=predicted_class_label,filename=filename)
    except Exception as e:
        return render_template('predict-image.html', error=str(e))

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        # Get the code snippet from the request
        code_snippet = request.form['code_snippet']

        # Tokenize the input text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([code_snippet])
        sequences = tokenizer.texts_to_sequences([code_snippet])

        max_sequence_length = 512
        padded_input = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

        expected_embedding_dimension = 768
        reshaped_input = tf.expand_dims(padded_input, axis=-1)
        reshaped_input = tf.repeat(reshaped_input, expected_embedding_dimension, axis=-1)

        # Make predictions using the loaded CodeBERT model
        predictions = codebertloaded_model.predict(reshaped_input)

        # Assuming predictions is a NumPy array, convert it to a list for JSON serialization
        predictions_list = predictions.tolist()

        # Map probabilities to class labels
        class_labels = ["O(logn)", "O(n^2)", "O(n^3)", "O(1)", "O(n)", "O(nlogn)", "Unknown", "O(2^n)"]
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        # Pass the variables to the template
        return render_template(
            'predict-text.html',
            code_snippet=code_snippet,
            class_label=predicted_class_label
        )
    except Exception as e:
        # Handle the exception and return an error response
        error_message = str(e)
        return render_template('predict-text.html', error=error_message), 500


if __name__ == '__main__':
    app.run(debug=True)
