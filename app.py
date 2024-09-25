from flask import Flask, render_template, request, jsonify
import os
import logging
from image_processing import process_image
from text_mining import analyze_text

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files or 'text' not in request.form:
            return jsonify({'error': 'Tidak ada gambar atau teks yang disediakan'}), 400
        
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)

        image_file = request.files['image']
        image_path = os.path.join(upload_folder, image_file.filename)
        image_file.save(image_path)
        processed_image = process_image(image_path)

        text = request.form['text']
        text_analysis = analyze_text(text)

        return jsonify({
            'image_result': processed_image,
            'text_result': text_analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
