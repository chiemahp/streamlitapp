# streamlitapp


# 🩺 Chest Disease Detector

A deep learning-powered application that detects chest diseases from X-ray images using a trained CNN model. Designed to support early diagnosis of pneumonia and other common thoracic conditions.

## 📌 Project Structure

chest_disease_predictor/
├── app.py # Streamlit web app for predictions
├── train_model.py # Script to train the CNN model
├── model/
│ └── chest_model.h5 # Trained Keras model
├── images/
│ └── sample_xray.jpg # Sample X-ray image for testing
├── README.md # Project documentation


## 🚀 Features

- Upload chest X-ray images to predict possible diseases
- Built with Convolutional Neural Networks (CNNs)
- Streamlit interface for easy web-based interaction
- High accuracy on pneumonia detection
- Option to retrain model with `train_model.py`

## 🧠 Model Details

- Framework: TensorFlow / Keras
- Architecture: CNN with multiple Conv2D and MaxPooling layers
- Input size: 224x224 grayscale chest X-ray images
- Trained on: [Kaggle Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/chest-disease-detector.git
cd chest-disease-detector

2. Install dependencies

pip install -r requirements.txt

3. Run the app

streamlit run app.py

🖼️ Sample Prediction

Upload a chest X-ray through the UI and the model will return a classification (e.g., Pneumonia, Normal).
🛠️ Training (Optional)

To retrain the model with new data:

python train_model.py

Ensure that the dataset is structured like:

chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
└── val/

✅ Requirements

    Python 3.7+

    TensorFlow

    Keras

    Streamlit

    OpenCV

    NumPy

    Matplotlib

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙋‍♀️ Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
✨ Acknowledgements

    Kaggle Chest X-ray Dataset

    Streamlit for UI

    TensorFlow/Keras for model building
