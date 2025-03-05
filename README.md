

A Flask-based application implementing Retrieval-Augmented Generation (RAG).

## Directory Structure

```
└── rag-flask-app/
    ├── app.py                # Main Flask application
    ├── requirements.txt      # List of dependencies
    ├── DATA/                 # Directory for storing dataset files
    ├── Notebook/
    │   └── Notebook.ipynb    # Jupyter Notebook for experimentation
    ├── src/
    │   └── helpers.py        # Helper functions for the application
    └── templates/
        └── index.html        # HTML template for the web interface
```

## Installation
### Clone the repository:
```sh
git clone https://github.com/yourusername/zabih1-rag-flask-app.git
cd rag-flask-app
```

### Create a virtual environment and activate it:
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the Flask application:
```sh
python app.py
```

