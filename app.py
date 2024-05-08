from flask import Flask, render_template, request, jsonify
import cv2
from rembg import remove
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# Function to remove background from an input image
def remove_background(input_image):
    # Convert the OpenCV image to RGB format for Rembg
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    try:
        # Remove the background using Rembg
        output_image_rgba = remove(input_image_rgb)

        # Convert the output back to the original color space (RGB)
        output_image_rgb = cv2.cvtColor(output_image_rgba, cv2.COLOR_RGBA2RGB)

        return output_image_rgb
    except Exception as e:
        print(f"Error encountered during background removal: {e}")
        return None

# Flask route to handle image processing
@app.route('/process', methods=['POST'])
def process():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'Empty file provided'}), 400
    
    # Read image file
    try:
        # Read the image file into a NumPy array
        input_image = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error reading image: {str(e)}'}), 400

    # Process the input image to remove the background
    processed_image = remove_background(input_image)

    if processed_image is None:
        return jsonify({'error': 'Error processing image'}), 500

    # Convert processed image to base64 for display
    buffered = io.BytesIO()
    Image.fromarray(processed_image).save(buffered, format="JPEG")
    img_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

    return jsonify({'image': img_str}), 200

# Flask route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
