#!/usr/bin/env python3
"""
AI Cancer Detection Web Application
Advanced Medical Imaging Analysis for Breast Cancer Histopathology

This Flask web application provides an interface for breast cancer detection
using a pre-trained CancerNet deep learning model.
"""

import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# TensorFlow will be imported only when needed
load_model = None
TENSORFLOW_AVAILABLE = False

app = Flask(__name__, 
           template_folder='cancerwebapp/templates',
           static_folder='cancerwebapp/static')
app.config['UPLOAD_FOLDER'] = 'cancerwebapp/uploads'
app.config['MODEL_PATH'] = 'CancerNet_working.h5'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'cancer-detection-app-secret-key-2024'

# Global variables
model = None
model_loading = False
model_loaded = False
model_load_start_time = None

# Class labels
CLASS_LABELS = {0: 'Benign', 1: 'Malignant'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_async():
    """Load model asynchronously."""
    global model, model_loading, model_loaded, model_load_start_time, load_model, TENSORFLOW_AVAILABLE
    
    model_loading = True
    model_load_start_time = time.time()
    
    try:
        print("Importing TensorFlow...")
        import tensorflow as tf
        from tensorflow.keras.models import load_model as tf_load_model
        load_model = tf_load_model
        TENSORFLOW_AVAILABLE = True
        print("TensorFlow imported successfully!")
        
        print("Loading CancerNet.h5 model...")
        
        # Simple loading for .h5 files (much more reliable than .keras)
        try:
            model = load_model(app.config['MODEL_PATH'], compile=False)
            print("✅ CancerNet.h5 model loaded successfully!")
            print(f"📊 Model input shape: {model.input_shape}")
            print(f"📊 Model output shape: {model.output_shape}")
            
            # Verify the model has proper trained weights
            weights = model.get_weights()
            if weights and len(weights) > 0:
                first_weights = weights[0]
                weight_range = first_weights.max() - first_weights.min()
                weight_std = first_weights.std()
                
                print(f"🔍 First layer weights range: {first_weights.min():.4f} to {first_weights.max():.4f}")
                print(f"🔍 Weight standard deviation: {weight_std:.4f}")
                
                # Check if weights look trained (should have reasonable variance)
                if weight_std > 0.01 and weight_range > 0.1:
                    print("✅ Model has proper trained weights!")
                    
                    # Quick functionality test
                    import numpy as np
                    test_input = np.random.random((1, 50, 50, 3)).astype('float32')
                    test_pred = model.predict(test_input, verbose=0)
                    print(f"🧪 Test prediction: {test_pred[0][0]:.4f}")
                    
                else:
                    print("⚠️ Warning: Model weights might be untrained or have issues")
                    print(f"   Weight std: {weight_std:.4f}, range: {weight_range:.4f}")
            else:
                print("❌ No weights found in model")
                raise Exception("Model has no weights")
                
        except FileNotFoundError:
            print(f"❌ Model file not found: {app.config['MODEL_PATH']}")
            print("   Make sure CancerNet.h5 exists in the current directory")
            raise Exception(f"Model file not found: {app.config['MODEL_PATH']}")
        except Exception as e:
            print(f"❌ Error loading CancerNet.h5: {e}")
            print("   Check if the .h5 file is valid and compatible")
            raise Exception(f"Could not load CancerNet.h5: {e}")
        
        # Recompile the model for predictions
        if model is not None:
            try:
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print("Model recompiled for predictions!")
            except Exception as compile_error:
                print(f"Warning: Could not recompile model: {compile_error}")
                print("Model will work for predictions without compilation")
        
        model_loaded = True
        print("Model loaded and ready for predictions!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
    finally:
        model_loading = False

def build_cancernet_model():
    """Rebuild the CancerNet model architecture."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
        
        model = Sequential()
        
        # Convolutional Block 1
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Convolutional Block 2
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Convolutional Block 3
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess image for prediction."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((50, 50))
        img = np.array(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

def validate_image_size(image_path):
    """Validate image dimensions."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return (width == 50 and height == 50), (width, height)
    except:
        return False, (0, 0)



def get_sample_files():
    """Get list of sample files from uploads directory."""
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        return {'benign': [], 'malignant': []}
    
    files = os.listdir(upload_dir)
    sample_files = {'benign': [], 'malignant': []}
    
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            if 'class0' in file:
                sample_files['benign'].append(file)
            elif 'class1' in file:
                sample_files['malignant'].append(file)
    
    return sample_files

def predict_image(image_path):
    """Make prediction on image."""
    global model
    
    if model is None:
        return None, 0.0, ["Model not loaded"]
    
    try:
        # Get filename for debugging
        filename = os.path.basename(image_path)
        expected_class = "class1 (Malignant)" if "class1" in filename else "class0 (Benign)" if "class0" in filename else "Unknown"
        
        print(f"\n🔍 PREDICTING: {filename}")
        print(f"📝 Expected: {expected_class}")
        
        # Preprocess image
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None, 0.0, ["Failed to preprocess image"]
        
        print(f"📊 Image shape: {img_array.shape}")
        print(f"📊 Pixel range: {img_array.min():.3f} to {img_array.max():.3f}")
        print(f"📊 Mean pixel value: {img_array.mean():.3f}")
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        print(f"🎯 Raw model output: {predictions}")
        
        # For binary classification with sigmoid output
        # predictions[0][0] gives the probability of class 1 (Malignant)
        probability_malignant = float(predictions[0][0])
        print(f"🎯 Probability of MALIGNANT: {probability_malignant:.4f}")
        print(f"🎯 Probability of BENIGN: {1-probability_malignant:.4f}")
        
        # Determine predicted class based on threshold of 0.5
        if probability_malignant > 0.5:
            predicted_class = 1  # Malignant
            confidence = probability_malignant
            prediction_text = "MALIGNANT"
        else:
            predicted_class = 0  # Benign
            confidence = 1.0 - probability_malignant
            prediction_text = "BENIGN"
        
        print(f"✅ PREDICTION: {prediction_text} (confidence: {confidence:.3f})")
        
        # Check if prediction matches expected
        is_correct = (
            ("class1" in filename and predicted_class == 1) or
            ("class0" in filename and predicted_class == 0)
        )
        if "class" in filename:
            print(f"{'✅ CORRECT' if is_correct else '❌ WRONG'} prediction!")
        
        # Get class label
        result = CLASS_LABELS[predicted_class]
        
        # Enhanced validation warnings
        warnings = []
        if confidence < 0.7:
            warnings.append("Low confidence prediction - consider getting a professional medical opinion")
        
        # Add debugging info if prediction seems wrong
        if not is_correct and "class" in filename:
            warnings.append(f"Debug: Expected {expected_class}, got {prediction_text}")
        
        print("─" * 50)
        return result, confidence, warnings
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, 0.0, [f"Prediction failed: {str(e)}"]

@app.route('/')
def index():
    """Main page route."""
    sample_files = get_sample_files()
    return render_template('index.html', 
                         sample_files=sample_files,
                         model_loaded=model_loaded)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Load model endpoint."""
    global model_loading, model_loaded
    
    if model_loaded:
        return jsonify({
            'status': 'already_loaded',
            'message': 'Model is already loaded and ready!'
        })
    
    if model_loading:
        return jsonify({
            'status': 'loading',
            'message': 'Model is currently being loaded...'
        })
    
    # Start loading model in background thread
    thread = threading.Thread(target=load_model_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Model loading started...'
    })

@app.route('/model_status')
def model_status():
    """Check model loading status."""
    global model_loading, model_loaded, model_load_start_time
    
    elapsed_time = 0
    if model_load_start_time:
        elapsed_time = time.time() - model_load_start_time
    
    if model_loaded:
        return jsonify({
            'loaded': True,
            'loading': False,
            'message': 'Model loaded successfully! Ready for predictions.',
            'elapsed': elapsed_time
        })
    elif model_loading:
        return jsonify({
            'loaded': False,
            'loading': True,
            'message': f'Loading model... Please wait.',
            'elapsed': elapsed_time
        })
    else:
        return jsonify({
            'loaded': False,
            'loading': False,
            'message': 'Model not loaded. Click "Load Model" to start.',
            'elapsed': 0
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if not model_loaded:
        if request.is_json or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
            return jsonify({'error': 'Please load the model first before making predictions.'})
        flash('Please load the model first before making predictions.', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        if request.is_json or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
            return jsonify({'error': 'No file selected. Please choose an image file.'})
        flash('No file selected. Please choose an image file.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        if request.is_json or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
            return jsonify({'error': 'No file selected. Please choose an image file.'})
        flash('No file selected. Please choose an image file.', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Validate image dimensions
            is_valid_size, dimensions = validate_image_size(file_path)
            validation_warnings = []
            
            if not is_valid_size:
                validation_warnings.append(f"Image size is {dimensions[0]}x{dimensions[1]} pixels, but model expects 50x50 pixels. Image will be resized, which may affect accuracy.")
            
            # Make prediction
            result, confidence, warnings = predict_image(file_path)
            
            if result is None:
                if request.is_json or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
                    return jsonify({'error': 'Failed to analyze image. Please try again with a different image.'})
                flash('Failed to analyze image. Please try again with a different image.', 'error')
                return redirect(url_for('index'))
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            # Combine all warnings
            all_warnings = validation_warnings + warnings
            
            # Check if this is an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
                return jsonify({
                    'success': True,
                    'result': result,
                    'confidence': f"{confidence:.1%}",
                    'filename': file.filename,
                    'warnings': all_warnings if all_warnings else None,
                    'validation_confidence': f"Image processed with {confidence:.1%} confidence" if is_valid_size else f"Image resized from {dimensions[0]}x{dimensions[1]} to 50x50 pixels"
                })
            
            return render_template('result.html',
                                 result=result,
                                 confidence=f"{confidence:.1%}",
                                 filename=file.filename,
                                 warnings=all_warnings if all_warnings else None,
                                 validation_confidence=f"Image processed with {confidence:.1%} confidence" if is_valid_size else f"Image resized from {dimensions[0]}x{dimensions[1]} to 50x50 pixels")
            
        except Exception as e:
            if request.is_json or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
                return jsonify({'error': f'Error processing image: {str(e)}'})
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        if request.is_json or request.headers.get('Content-Type', '').startswith('multipart/form-data'):
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'})
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
        return redirect(url_for('index'))

@app.route('/predict_sample/<filename>', methods=['POST'])
def predict_sample(filename):
    """Predict using sample image."""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please load the model first.'})
    
    sample_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(sample_path):
        flash('Sample image not found.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Determine expected result from filename
        if 'class0' in filename:
            expected = 'Benign'
        elif 'class1' in filename:
            expected = 'Malignant'
        else:
            expected = 'Unknown'
        
        # Validate image dimensions
        is_valid_size, dimensions = validate_image_size(sample_path)
        
        # Make prediction
        result, confidence, warnings = predict_image(sample_path)
        
        if result is None:
            flash('Failed to analyze sample image.', 'error')
            return redirect(url_for('index'))
        
        return render_template('result.html',
                             result=result,
                             confidence=f"{confidence:.1%}",
                             filename=filename,
                             expected=expected,
                             is_sample=True,
                             warnings=warnings if warnings else None,
                             validation_confidence=f"Sample image processed with {confidence:.1%} confidence")
        
    except Exception as e:
        flash(f'Error processing sample: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    flash('An internal server error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("=" * 60)
    print("🧠 AI Cancer Detection System")
    print("Advanced Medical Imaging Analysis for Breast Cancer Histopathology")
    print("=" * 60)
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("🤖 Model file: CancerNet_working.h5")
    print("🌐 Starting Flask application...")
    print("=" * 60)
    
    # Run the Flask app
    # For production (Render), use gunicorn. For local development, use Flask dev server
    if os.environ.get('RENDER'):
        # Production mode - let gunicorn handle the app
        pass
    else:
        # Development mode
        app.run(debug=True, host='0.0.0.0', port=5000)