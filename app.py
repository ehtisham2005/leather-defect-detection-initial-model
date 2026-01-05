from flask import Flask, request, redirect, render_template_string
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
# Import the specific function for ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# --- CONFIG ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. LOAD MODEL ---
print("Loading model...")
try:
    # We load the model and tell Keras about the custom preprocessing function
    model = tf.keras.models.load_model(
        'leather_defect_model.h5', 
        custom_objects={'preprocess_input': preprocess_input}
    )
except Exception as e:
    print(f"Standard load failed: {e}. Trying safe mode...")
    model = tf.keras.models.load_model('leather_defect_model.h5', safe_mode=False)

# Load class names or use default fallback
try:
    with open("class_names.txt", "r") as f:
        class_names = f.read().splitlines()
except:
    class_names = ['Folding marks', 'Grain off', 'Growth marks', 'loose grains', 'non defective', 'pinhole']

print("Model loaded successfully.")

# --- 2. THE FRONTEND (HTML + CSS + JS) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leather QC | AI Inspector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #2563eb; --success: #16a34a; --danger: #dc2626; --bg: #f8fafc; --card-bg: #ffffff; --text-main: #1e293b; }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text-main); margin: 0; padding: 40px 20px; display: flex; justify-content: center; min-height: 100vh; }
        .container { width: 100%; max-width: 900px; background: var(--card-bg); padding: 40px; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.05); }
        h1 { font-weight: 800; font-size: 2.5rem; margin: 0 0 10px 0; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
        .subtitle { color: #64748b; font-size: 1.1rem; text-align: center; margin-bottom: 40px; }
        
        /* Upload Area */
        .upload-area { border: 2px dashed #cbd5e1; border-radius: 15px; padding: 50px 20px; text-align: center; cursor: pointer; transition: all 0.3s ease; background-color: #f1f5f9; display: block; }
        .upload-area:hover, .upload-area.dragover { border-color: var(--primary); background-color: #eff6ff; transform: translateY(-2px); }
        .upload-icon { font-size: 3rem; margin-bottom: 15px; display: block; }
        .upload-text { color: #64748b; font-weight: 600; }

        /* Results */
        .verdict-banner { margin: 30px 0; padding: 20px; border-radius: 12px; text-align: center; font-weight: 800; font-size: 1.5rem; letter-spacing: 1px; text-transform: uppercase; }
        .verdict-reject { background: #fef2f2; color: var(--danger); border: 1px solid #fee2e2; }
        .verdict-accept { background: #f0fdf4; color: var(--success); border: 1px solid #dcfce7; }

        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 25px; margin-top: 30px; }
        .card { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; }
        .card img { width: 100%; height: 200px; object-fit: cover; border-bottom: 1px solid #f1f5f9; }
        .card-body { padding: 15px; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
        .badge-bad { background: #fee2e2; color: var(--danger); }
        .badge-good { background: #dcfce7; color: var(--success); }
        .defect-name { font-weight: 600; font-size: 1rem; margin-bottom: 5px; }
        .confidence-label { font-size: 0.8rem; color: #64748b; display: flex; justify-content: space-between; margin-bottom: 5px; }
        .progress-bar { height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 3px; }
        
        .btn-reset { display: block; width: 100%; padding: 15px; margin-top: 30px; background: #1e293b; color: white; border: none; border-radius: 10px; font-weight: 600; cursor: pointer; text-decoration: none; text-align: center; }
        .btn-reset:hover { background: #0f172a; }
    </style>
</head>
<body>
<div class="container">
    <div>
        <h1>👜 Quality Control AI</h1>
        <div class="subtitle">Automated Leather Defect Detection System</div>
    </div>

    <form id="upload-form" method="POST" enctype="multipart/form-data">
        <label id="drop-zone" class="upload-area">
            <span class="upload-icon">📸</span>
            <span class="upload-text">Upload Close-Up Texture Photos</span>
            <input type="file" name="files[]" id="file-input" multiple required onchange="this.form.submit()" hidden>
        </label>
    </form>

    {% if results %}
        {% if bag_rejected %}
            <div class="verdict-banner verdict-reject">❌ REJECT BAG</div>
        {% else %}
            <div class="verdict-banner verdict-accept">✅ ACCEPT BAG</div>
        {% endif %}

        <div class="grid">
            {% for res in results %}
            <div class="card">
                <img src="{{ url_for('static', filename='uploads/' + res.image) }}" alt="bag">
                <div class="card-body">
                    {% if res.is_defect %}
                        <span class="badge badge-bad">Defect</span>
                    {% else %}
                        <span class="badge badge-good">Pass</span>
                    {% endif %}
                    <div class="defect-name">{{ res.class }}</div>
                    <div class="confidence-label">
                        <span>Confidence</span><span>{{ res.confidence }}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ res.confidence }}%; background: {% if res.is_defect %}var(--danger){% else %}var(--success){% endif %}"></div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        <a href="/" class="btn-reset">Inspect Next Bag</a>
    {% endif %}
</div>

<script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const form = document.getElementById('upload-form');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            form.submit();
        }
    });
</script>

</body>
</html>
"""

# --- 3. PREDICTION LOGIC ---
def predict_image(image_path):
    img = Image.open(image_path)
    
    # --- AUTO-ZOOM FIX ---
    # Since dataset images are pure texture (no background),
    # we crop the center square of your photo to zoom in.
    width, height = img.size
    new_size = min(width, height)
    
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    
    img = img.crop((left, top, right, bottom))
    # ---------------------

    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Note: Preprocessing is handled inside the model thanks to our "Clean Version" fix.
    # No manual preprocess_input needed here!

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files[]')
        results = []
        bag_rejected = False

        for file in files:
            if file.filename == '': continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            pred_class, confidence = predict_image(filepath)
            
            # Logic
            is_defect = pred_class != "non defective"
            if is_defect: bag_rejected = True
            
            results.append({
                'image': filename,
                'class': pred_class,
                'confidence': round(confidence, 2),
                'is_defect': is_defect
            })
        return render_template_string(HTML_TEMPLATE, results=results, bag_rejected=bag_rejected)

    return render_template_string(HTML_TEMPLATE, results=None)

if __name__ == '__main__':
    app.run(debug=True)