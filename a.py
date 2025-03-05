import os
from flask import Flask, render_template, request, jsonify
import werkzeug
from src.helpers import setup_vector_database, create_rag_chain

app = Flask(__name__)
app.config['DATA'] = 'DATA'
os.makedirs(app.config['DATA'], exist_ok=True)

current_vector_db = None
current_rag_chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handles PDF file upload, processes it, and initializes the RAG pipeline."""
    global current_vector_db, current_rag_chain
    
    if 'pdf-file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['pdf-file']
    
    if file.filename == '':
        return "No selected file", 400
    
    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join(app.config['DATA'], filename)
    file.save(filepath)
    
    current_vector_db = setup_vector_database(filepath)
    current_rag_chain = create_rag_chain(current_vector_db)
    
    return jsonify({'message': 'PDF uploaded successfully'}), 200

@app.route('/ask', methods=['POST'])
def ask():
    """Handles user queries and returns responses from the RAG pipeline."""
    global current_rag_chain
    
    if current_rag_chain is None:
        return jsonify({'answer': "Please upload a PDF first before asking questions."}), 400
    
    message = request.form.get('messageText', '')
    response = current_rag_chain.invoke(message)
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)
