---
title: Handwriting Recognizer using CNN and Streamlit
---

This project is a web-based Handwritten Digit Recognizer built using a
Convolutional Neural Network (CNN) trained on the MNIST dataset. The
model is deployed with a Streamlit frontend for easy interaction.

# 🚀 Demo

Live App: https://handwrintingrecognizer.streamlit.app

# 📂 Project Structure

├── app.py \# Main Streamlit application\
├── main.py \# Model training script\
├── mnist_cnn_model.h5 \# Pretrained CNN model\
├── requirements.txt \# Python dependencies\
├── runtime.txt \# Python version for deployment\
├── explanation/ \# Additional documentation (if any)\
└── .gitignore \# Ignores virtual environment, logs, etc.

# 🧠 Model Details

\- Dataset: MNIST (http://yann.lecun.com/exdb/mnist/)\
- Architecture: Convolutional Neural Network (CNN)\
- Framework: TensorFlow/Keras\
- Training:\
- Input shape: (28x28 grayscale images)\
- Output: 10 classes (digits 0--9)

# 🖥️ How to Run Locally

1\. Clone the repository\
git clone https://github.com/CharanChandu-11/handwrintingrecognizer.git\
cd handwrintingrecognizer\
\
2. Create a virtual environment (optional but recommended)\
python -m venv venv\
source venv/bin/activate \# On Windows use venv\\Scripts\\activate\
\
3. Install dependencies\
pip install -r requirements.txt\
\
4. Run the app\
streamlit run app.py

# ☁️ Deployment (Streamlit Cloud)

\- Ensure you have a runtime.txt with the line:\
python-3.10\
\
- Your requirements.txt should include:\
streamlit\>=1.28.0\
tensorflow\
numpy\>=1.21.0\
matplotlib\>=3.5.0\
Pillow\>=8.3.0\
\
- Push your code to GitHub and connect the repo to Streamlit Cloud.

# ✨ Features

\- Draw a digit on canvas\
- Real-time prediction\
- Clean and minimal UI\
- Fully serverless deployment

# 🙌 Acknowledgements

\- MNIST Dataset - Yann LeCun (http://yann.lecun.com/exdb/mnist/)\
- TensorFlow / Keras\
- Streamlit Cloud for free hosting

# 📬 Contact

Built with ❤️ by Charan Chandu (https://github.com/CharanChandu-11)\
For feedback or collaboration, feel free to open an issue or pull
request!
