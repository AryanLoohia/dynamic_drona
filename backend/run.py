import os
from webapp import app

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Create runs/detect directory if it doesn't exist
    if not os.path.exists('runs/detect'):
        os.makedirs('runs/detect')
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=True) 