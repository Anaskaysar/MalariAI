import os
import uuid
from flask import Flask, request, jsonify, render_template
from pipeline import MalariAI

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model
CHECKPOINT = "checkpoints/best.pth"
if not os.path.exists(CHECKPOINT):
    print(f"WARNING: Checkpoint {CHECKPOINT} not found. Running with ImageNet weights.")
    model = MalariAI()
else:
    model = MalariAI(CHECKPOINT)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    # Save temp file
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Run pipeline
        results = model.analyze(filepath)
        
        # Cleanup temp file
        # os.remove(filepath) # Keep it if we want to serve it, but we use base64 now
        
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # HF Spaces expects port 7860
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
