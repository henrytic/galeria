import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from processor import FaceProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

processor = FaceProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    saved_files = []
    
    for file in files:
        if file.filename == '':
            continue
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_files.append(filename)
    
    # Trigger processing
    # In a real app, this should be async (Celery/RQ)
    # For this demo, we run it synchronously
    try:
        people_summary = processor.process_images()
        return jsonify({'message': 'Files uploaded and processed', 'people': people_summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gallery')
def get_gallery():
    images = processor.get_gallery_images()
    return jsonify({'images': images})

@app.route('/api/people')
def get_people():
    # If the server restarted, we might need to re-process or load state.
    # For this demo, we assume process_images was called at least once or state is in processor.
    # Note: processor.process_images() re-runs everything. 
    # processor.get_people_summary() returns current state.
    # If empty, maybe run process?
    if not processor.data_records and os.listdir(app.config['UPLOAD_FOLDER']):
         processor.process_images()
         
    return jsonify(processor.get_people_summary())

@app.route('/api/person/<id>')
def get_person(id):
    images = processor.get_person_images(id)
    return jsonify({'images': images})

@app.route('/api/metrics')
def get_metrics():
    metrics = processor.get_metrics()
    return jsonify(metrics)

@app.route('/api/scatter')
def get_scatter():
    data = processor.get_scatter_data()
    return jsonify(data)

@app.route('/api/person/<id>/rename', methods=['POST'])
def rename_person(id):
    data = request.get_json()
    new_name = data.get('name')
    if new_name:
        processor.rename_person(id, new_name)
        return jsonify({'success': True})
    return jsonify({'error': 'Name is required'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
