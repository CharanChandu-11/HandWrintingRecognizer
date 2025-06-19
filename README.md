✍️ Handwriting Digit Recognizer
A web-based digit recognition app built using Streamlit and trained with the MNIST dataset using TensorFlow. Draw any digit (0–9) and get instant predictions powered by a Convolutional Neural Network (CNN).

📌 Features
🎨 Draw digits directly on a canvas

🔍 Predicts digits using a trained CNN model

📊 Real-time feedback and visualization

🧠 Built with TensorFlow and Streamlit

🌐 Deployable as a web app

🖥️ Demo
🧪 Live demo: handwrintingrecognizer.streamlit.app
(Make sure to try drawing digits in the canvas and clicking "Predict")

📁 Project Structure
HandWritingRecognizer/
│
├── app.py                # Streamlit app entry point
├── model.h5              # Trained CNN model file
├── requirements.txt      # Python dependencies
└── utils.py              # Helper functions for preprocessing and prediction
🚀 Installation
Option 1: Run Locally
bash
Copy
Edit
git clone https://github.com/CharanChandu-11/handwrintingrecognizer.git
cd handwrintingrecognizer
pip install -r requirements.txt
streamlit run app.py
Option 2: Run on Streamlit Cloud
Push to a public GitHub repo.

Deploy via streamlit.io/cloud.

🧠 Model Details
Dataset: MNIST (handwritten digit images)

Model Type: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Input Shape: (28, 28, 1)

Accuracy: ~99% on test data

📦 Requirements
streamlit>=1.28.0
tensorflow
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.3.0
🧰 Usage
Draw any digit (0–9) using the canvas on the app.

Click the "Predict" button.

See the prediction instantly along with confidence scores.

📸 Screenshot

Example output after predicting handwritten "8"

🧑‍💻 Author
Charan Chandu
🔗 www.linkedin.com/in/p-charanchandu-74951030b
🌐 GitHub

📜 License
This project is open-sourced under the MIT License.
