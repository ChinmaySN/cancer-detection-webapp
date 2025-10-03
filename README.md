# Cancer Detection Web Application

🧬 **AI-Powered Breast Cancer Detection System**

An advanced web application that uses deep learning to analyze breast cancer histopathology images and classify them as benign or malignant.

## 🚀 Features

- **AI-Powered Analysis**: Uses a trained CancerNet deep learning model
- **Web Interface**: User-friendly Flask web application
- **Sample Testing**: Pre-loaded sample images for testing
- **High Accuracy**: Trained model with validated performance
- **Real-time Predictions**: Instant analysis results

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **AI/ML**: TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: PIL (Pillow)
- **Model**: CancerNet CNN Architecture

## 🏃‍♂️ Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/ChinmaySN/cancer-detection-webapp.git
   cd cancer-detection-webapp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://localhost:5000
   ```

### Usage

1. **Load Model**: Click "Load Model" button to initialize the AI model
2. **Upload Image**: Upload a histopathology image (PNG, JPG, JPEG)
3. **Get Results**: View classification results with confidence scores
4. **Test Samples**: Use provided sample images to test the system

## 🏗️ Deployment

### Deploy on Render

1. **Connect GitHub**: Link your repository to Render
2. **Configure**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: 3.11.9

## 📁 Project Structure

```
cancer-detection-webapp/
├── app.py                          # Main Flask application
├── CancerNet_working.h5            # Trained model file
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── cancerwebapp/
    ├── static/
    │   └── style.css              # CSS styles
    ├── templates/
    │   ├── index.html             # Main page template
    │   └── result.html            # Results page template
    └── uploads/
        └── (sample images)        # Sample test images
```

## 🧠 Model Information

- **Architecture**: CancerNet CNN
- **Input**: 50x50 RGB histopathology images
- **Output**: Binary classification (Benign/Malignant)
- **Framework**: TensorFlow/Keras
- **File**: CancerNet_working.h5

## ⚠️ Medical Disclaimer

This application is for **research and educational purposes only**. It is **not intended for medical diagnosis**. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

**⚡ Built with TensorFlow and Flask | 🧬 Advancing AI in Healthcare**