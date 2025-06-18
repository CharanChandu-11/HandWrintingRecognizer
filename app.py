import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image, ImageDraw
import io
import base64
from streamlit_drawable_canvas import st_canvas

# Configure page
st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the pre-trained MNIST model"""
    try:
        # Try to load your saved model first
        model = keras.models.load_model('mnist_cnn_model.h5')
        return model, "Custom Model"
    except:
        try:
            # Fallback: create and train a simple model
            st.info("Custom model not found. Loading a simple pre-trained model...")
            model = create_simple_model()
            return model, "Simple Model"
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, "No Model"

def create_simple_model():
    """Create and train a simple MNIST model if the main model isn't available"""
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Create simple model
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Quick training (just 3 epochs for demo)
    with st.spinner("Training a simple model (this will take a moment)..."):
        model.fit(x_train[:10000], y_train[:10000], epochs=3, verbose=0)
    
    return model

def preprocess_canvas_image(canvas_data):
    """Convert canvas drawing to 28x28 grayscale image suitable for MNIST model"""
    if canvas_data is None or canvas_data.image_data is None:
        return None
    
    # Get the image data
    img_data = canvas_data.image_data
    
    # Convert to PIL Image
    img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert colors (canvas is black on white, MNIST is white on black)
    img_array = 255 - img_array
    
    # Find bounding box of the drawing
    coords = np.column_stack(np.where(img_array > 50))  # Find non-zero pixels
    if len(coords) == 0:
        return None
    
    # Get bounding box
    y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
    x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Add some padding
    padding = 20
    y_min = max(0, y_min - padding)
    y_max = min(img_array.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(img_array.shape[1], x_max + padding)
    
    # Crop to bounding box
    cropped = img_array[y_min:y_max, x_min:x_max]
    
    # Resize to 28x28 while maintaining aspect ratio
    h, w = cropped.shape
    if h > w:
        new_h, new_w = 20, int(20 * w / h)
    else:
        new_h, new_w = int(20 * h / w), 20
    
    # Resize
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create 28x28 canvas and center the digit
    canvas_28 = np.zeros((28, 28), dtype=np.uint8)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas_28[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    # Normalize to [0, 1]
    canvas_28 = canvas_28.astype('float32') / 255.0
    
    # Reshape for model input
    canvas_28 = canvas_28.reshape(1, 28, 28, 1)
    
    return canvas_28

def predict_digit(model, processed_image):
    """Make prediction on the processed image"""
    if processed_image is None or model is None:
        return None, None, None
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = int(np.argmax(prediction[0]))  # Convert to Python int
    confidence = float(np.max(prediction[0]))  # Convert to Python float
    all_predictions = prediction[0].astype(float).tolist()  # Convert to Python list
    
    return predicted_class, confidence, all_predictions

# Main app
def main():
    st.title("🔢 MNIST Digit Predictor")
    st.markdown("Draw a digit (0-9) on the canvas below and see the AI prediction!")
    
    # Load model
    model, model_type = load_model()
    
    if model is None:
        st.error("Could not load model. Please ensure 'mnist_cnn_model.h5' is in the same directory.")
        return
    
    st.success(f"✅ {model_type} loaded successfully!")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Draw Here")
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
            stroke_width=15,
            stroke_color="black",
            background_color="white",
            background_image=None,
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            point_display_radius=0,
            key="canvas",
        )
        
        # Control buttons
        col_clear, col_predict = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.rerun()
        
        with col_predict:
            predict_button = st.button("🔮 Predict", use_container_width=True, type="primary")
    
    with col2:
        st.subheader("Prediction Results")
        
        if canvas_result.image_data is not None and (predict_button or st.session_state.get('auto_predict', False)):
            # Process the image
            processed_image = preprocess_canvas_image(canvas_result)
            
            if processed_image is not None:
                # Make prediction
                predicted_digit, confidence, all_predictions = predict_digit(model, processed_image)
                
                if predicted_digit is not None:
                    # Display prediction
                    st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Show processed image
                    st.subheader("Processed Image (28x28)")
                    processed_display = processed_image.reshape(28, 28)
                    st.image(processed_display, width=140, use_column_width=False)
                    
                    # Show all predictions
                    st.subheader("All Predictions")
                    for i, prob in enumerate(all_predictions):
                        st.write(f"Digit {i}: {prob:.3f} ({prob:.1%})")
                    
                    # Prediction bar chart
                    st.subheader("Prediction Probabilities")
                    chart_data = {
                        'Digit': [str(i) for i in range(10)],
                        'Probability': all_predictions
                    }
                    st.bar_chart(data=chart_data, x='Digit', y='Probability')
                else:
                    st.error("Could not make prediction. Please try drawing again.")
            else:
                st.warning("No drawing detected. Please draw a digit on the canvas.")
        else:
            st.info("👈 Draw a digit on the canvas and click 'Predict' to see the results!")
    
    # Settings sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Auto-predict option
        auto_predict = st.checkbox("Auto-predict while drawing", value=False)
        st.session_state['auto_predict'] = auto_predict
        
        # Model info
        st.header("📊 Model Info")
        st.write(f"**Model Type:** {model_type}")
        if model is not None:
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Total Parameters:** {model.count_params():,}")
        
        # Instructions
        st.header("📝 Instructions")
        st.markdown("""
        1. **Draw** a digit (0-9) on the white canvas
        2. **Click Predict** to see the AI's guess
        3. **Clear Canvas** to start over
        
        **Tips:**
        - Draw digits clearly and centered
        - Use thick strokes for better recognition
        - Try different writing styles!
        """)
        
        # Sample predictions (if you want to show examples)
        st.header("🎯 Tips for Better Accuracy")
        st.markdown("""
        - **Size**: Draw digits that fill most of the canvas
        - **Thickness**: Use thick, bold strokes
        - **Center**: Keep digits centered in the canvas
        - **Style**: Write digits clearly, similar to how you'd write them normally
        """)

if __name__ == "__main__":
    main()