# 🔢 Digit Recognition Web App
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://digit-recognition-app-latest.onrender.com)


A modern web application that uses deep learning to recognize handwritten digits (0-9). Built with Flask and TensorFlow, featuring both image upload and drawing canvas functionality.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-red.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **📁 Image Upload**: Upload PNG/JPG images of handwritten digits
- **✏️ Drawing Canvas**: Draw digits directly in the browser with touch support
- **🧠 AI Prediction**: Real-time digit recognition using trained neural network
- **📊 Confidence Scores**: View prediction probabilities for all digits (0-9)
- 🐳 **Dockerized Deployment**: Run the app in a container with all dependencies pre-installed. Easy to deploy locally or on cloud platforms like Render, Heroku, or AWS.
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🎨 Modern UI**: Beautiful gradient interface with smooth animations

## 🚀 Demo

### Image Upload
Upload any image containing a handwritten digit and get instant predictions:

### Drawing Canvas  
Draw digits directly in the browser:

### Results Display
View predictions with confidence scores and probability distributions:

## 🌐 Live Demo

Try the app online without any setup!  
[Click here to use the Digit Recognition Web App](https://digit-recognition-app-latest.onrender.com)


## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pushkar-112/DL-Handwritten-Digit-Recognition.git
   cd DL-Handwritten-Digit-Recognition
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your trained model**
   - Place your trained model file (`*.h5`) in the `model/` directory
   - Rename it to `digit_recognition_model.h5` OR update the path in `app_production.py`

5. **Run the application**
   ```bash
   python app_production.py
   ```

6. **Open your browser** and navigate to `http://localhost:5000`

## 📁 Project Structure

```
DL-Handwritten-Digit-Recognition/
├── app_production.py          # Main Flask application (production)
├── app.py                     # Development version with debug features
├── requirements.txt           # Python dependencies
├── model/
│   └── digit_recognition_model.h5   # Your trained model (add this file)
├── templates/
│   ├── index_production.html  # Clean production template
│   └── index.html            # Debug template
├── debug_model_path.py       # Model path debugging utility
├── quick_fix.py              # Automatic path fixing utility
└── README.md                 # This file
```

## 🔧 Configuration

### Model Requirements
Your model should:
- Accept flattened 28×28 grayscale images (784 features)
- Input shape: `(None, 784)`
- Output 10 classes (digits 0-9)
- Be saved as a `.h5` file (Keras format)

### Supported Model Architecture Example
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

## 🎯 Usage

### Image Upload
1. Click "Upload Image" section
2. Select a PNG/JPG file containing a handwritten digit
3. Click "🔮 Predict from Image"
4. View results with confidence scores

### Drawing Canvas
1. Use mouse or touch to draw a digit in the canvas
2. Click "🔮 Predict from Drawing" 
3. Use "🗑️ Clear Canvas" to reset and try again

## 🧪 Development

### Debug Mode
For development and troubleshooting, use the debug version:
```bash
python app.py
```

This includes additional features:
- Image preprocessing visualization
- Debug endpoint (`/debug_image`)
- Detailed error logging
- Step-by-step preprocessing display

### Model Path Issues
If you encounter model loading issues:
```bash
python debug_model_path.py  # Diagnose path problems
python quick_fix.py         # Auto-fix common path issues
```

## 🔍 API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Prediction endpoint
  - Accepts: Form data with image file OR JSON with canvas data
  - Returns: JSON with predicted digit and probabilities
- `GET /health` - Health check endpoint

### Example API Usage
```python
import requests

# Upload image
with open('digit_image.png', 'rb') as f:
    response = requests.post('http://localhost:5000/predict', 
                           files={'file': f})
    result = response.json()
    print(f"Predicted digit: {result['predicted_digit']}")
```

## 🚀 Deployment

### Local Development
```bash
python app_production.py
```

### Production Deployment
The app is ready for deployment on platforms like:
- **Heroku**: Add `Procfile` with `web: python app_production.py`
- **AWS EC2**: Use gunicorn with `gunicorn app_production:app`
- **Docker**: Create Dockerfile based on Python slim image

### Environment Variables (Optional)
```bash
export FLASK_ENV=production
export MODEL_PATH=model/your_model.h5
```

## 🧠 How It Works

### Image Preprocessing Pipeline
1. **Resize** → 28×28 pixels (MNIST standard)
2. **Grayscale** → Convert RGB to single channel
3. **Background Detection** → Auto-invert if light background detected
4. **Enhancement** → Increase contrast and apply thresholding
5. **Normalization** → Scale pixel values to 0-1 range
6. **Flattening** → Reshape to (1, 784) for model input

### Model Architecture
The app expects a trained model with:
- Dense layers (not convolutional)
- Input: Flattened 28×28 images (784 features)
- Output: 10 classes with softmax activation

## 🧠 Model Details - digit_recognition_model.h5

- **Dataset**: MNIST handwritten digits
- **Architecture**: 
  - Input: Flattened 28x28 images (784 features)
  - Dense Layer 1: 128 units, ReLU
  - Dense Layer 2: 10 units, Softmax
- **Training**: 
  - Optimizer: Adam
  - Loss: Sparse Categorical Crossentropy
  - Epochs: 10
  - Validation split: 0.1
  - Scaling applied: pixel values normalized to [0,1]
- **Overfitting Mitigation**: Dropout layers used
- **Test Accuracy**: ~98%
- **Notebook**: [Link to the Jupyter notebook](notebooks\HandwrittenDigitRecognitionNN.ipynb)


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/) web framework
- Neural network powered by [TensorFlow](https://tensorflow.org/)
- Inspired by the classic MNIST dataset
- UI designed with modern CSS gradients and animations

## 📞 Support

If you encounter any issues:

1. **Model Loading Problems**: Run `python debug_model_path.py`
2. **Path Issues**: Run `python quick_fix.py`
3. **General Issues**: Check the console output for error messages
4. **Feature Requests**: Open an issue on GitHub

## 📈 Future Enhancements

- [ ] Support for multi-digit numbers
- [ ] Batch image processing
- [ ] Model confidence thresholding
- [ ] Export prediction results
- [ ] Additional pre-trained models
- [ ] REST API documentation
- [ ] Docker containerization
- [ ] Performance metrics dashboard

---

⭐ **Star this repository if you found it helpful!**

**Made with ❤️ and Python** 🐍