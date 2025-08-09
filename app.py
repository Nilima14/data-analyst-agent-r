import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.middleware.proxy_fix import ProxyFix
from data_analyst import DataAnalyst
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize data analyst
data_analyst = DataAnalyst()

@app.route('/')
def index():
    """Serve a simple test interface"""
    return render_template('index.html')

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis"""
    try:
        # Check if questions.txt is present
        if 'questions.txt' not in request.files:
            return jsonify({'error': 'questions.txt file is required'}), 400
        
        questions_file = request.files['questions.txt']
        if questions_file.filename == '':
            return jsonify({'error': 'questions.txt file is required'}), 400
        
        # Read the questions
        questions_content = questions_file.read().decode('utf-8')
        
        # Get additional files
        additional_files = {}
        for file_key in request.files:
            if file_key != 'questions.txt':
                file_obj = request.files[file_key]
                if file_obj.filename != '':
                    additional_files[file_key] = {
                        'content': file_obj.read(),
                        'filename': file_obj.filename,
                        'content_type': file_obj.content_type
                    }
        
        # Process the analysis request
        result = data_analyst.analyze(questions_content, additional_files)
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in analyze_data: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
