import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from tensorflow.keras import layers, models, regularizers
import tempfile
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set page configuration
st.set_page_config(
    page_title="Kidney Stone Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.2rem;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .stone {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .training-progress {
        background-color: #e9ecef;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class KidneyStoneModel:
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.input_dim = img_size[0] * img_size[1] * 3
        self.class_names = ['Normal', 'Stone']
        self.models = {}

    def create_dnn_model(self):
        """Create Deep Neural Network model"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.3),
            layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.3),
            layers.Dense(2, activation="softmax")
        ])

        model.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def create_mlp_model(self):
        """Create MLP model"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.2),
            layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.2),
            layers.Dense(2, activation="softmax")
        ])

        model.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def create_autoencoder_dnn_model(self):
        """Create Autoencoder + DNN model"""
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = layers.Dense(512, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        latent = layers.Dense(128, activation="relu")(x)
        y = layers.Dense(256, activation="relu")(latent)
        y = layers.Dense(512, activation="relu")(y)
        decoded = layers.Dense(self.input_dim, activation="sigmoid")(y)

        # Classifier
        classifier_output = layers.Dense(2, activation="softmax")(latent)
        classifier = tf.keras.Model(inputs, classifier_output)
        classifier.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return classifier

    def load_and_preprocess_data(self, data_dir="dataset"):
        """Load and preprocess the kidney stone dataset"""
        try:
            # Check if dataset exists
            if not os.path.exists(data_dir):
                st.error(f"❌ Dataset directory '{data_dir}' not found!")
                st.info("""
                Please upload a zip file containing your dataset in the 'Upload Dataset' section.
                The dataset should have this structure:
                ```
                dataset/
                ├── Normal/
                │   ├── image1.jpg
                │   ├── image2.jpg
                │   └── ...
                └── Stone/
                    ├── image1.jpg
                    ├── image2.jpg
                    └── ...
                ```
                """)
                return None, None, None, None

            # Load datasets
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=self.img_size,
                batch_size=32
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=self.img_size,
                batch_size=32
            )

            # Get class names
            self.class_names = train_ds.class_names
            st.success(f"✅ Found classes: {self.class_names}")

            # Cache/prefetch for performance
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            # Normalize and flatten
            normalization_layer = layers.Rescaling(1. / 255)

            def preprocess_ds(ds):
                return ds.map(lambda x, y: (tf.reshape(normalization_layer(x), [tf.shape(x)[0], -1]), y))

            train_ds_flat = preprocess_ds(train_ds)
            val_ds_flat = preprocess_ds(val_ds)

            return train_ds_flat, val_ds_flat, self.class_names

        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return None, None, None, None

    def train_models(self, train_ds, val_ds, epochs=10):
        """Train all models on real data"""
        models_info = {
            'DNN': self.create_dnn_model(),
            'MLP': self.create_mlp_model(),
            'AE-DNN': self.create_autoencoder_dnn_model()
        }

        results = {}

        for model_name, model in models_info.items():
            st.write(f"**Training {model_name} model...**")

            # Create progress bar for this model
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Train model
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=0
            )

            # Evaluate model
            val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

            # Store results
            results[model_name] = {
                'model': model,
                'history': history.history,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss
            }

            # Save model
            model_path = f"{model_name.lower()}_model.keras"
            model.save(model_path)
            self.models[model_name] = model

            # Update progress
            progress_bar.progress(100)
            status_text.success(f"✅ {model_name} trained - Accuracy: {val_accuracy:.2%}")

        return results

    def predict_image(self, image, model_name='MLP'):
        """Predict kidney stone from image"""
        if model_name not in self.models:
            st.error(f"Model {model_name} not found. Please train models first.")
            return None, None, None

        model = self.models[model_name]

        # Preprocess image
        img = image.resize(self.img_size)
        img_array = np.array(img) / 255.0

        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]

        img_array = img_array.reshape(1, -1)  # Flatten

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        class_name = self.class_names[predicted_class]
        probabilities = {
            'Normal': predictions[0][0],
            'Stone': predictions[0][1]
        }

        return class_name, confidence, probabilities


def extract_zip_file(zip_file):
    """Extract uploaded zip file to dataset directory"""
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        # Save uploaded zip file
        zip_path = os.path.join(temp_dir, "uploaded_dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getvalue())

        # Extract zip file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset")

        st.success("✅ Dataset extracted successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Error extracting zip file: {str(e)}")
        return False


def main():
    # Initialize model handler
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = KidneyStoneModel()

    model_handler = st.session_state.model_handler

    # Header
    st.markdown('<h1 class="main-header">🩺 Kidney Stone Classification System</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode",
                                    ["🏠 Home", "📁 Upload Dataset", "🔄 Train Models", "🔍 Predict", "📊 Model Info"])

    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "📁 Upload Dataset":
        upload_dataset_interface()
    elif app_mode == "🔄 Train Models":
        train_models_interface(model_handler)
    elif app_mode == "🔍 Predict":
        predict_interface(model_handler)
    elif app_mode == "📊 Model Info":
        show_model_info(model_handler)


def show_home():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Welcome to the Kidney Stone Classification System

        This AI-powered application helps classify kidney ultrasound images into:
        - **Normal** - No kidney stones detected
        - **Stone** - Kidney stones present

        ### 🚀 Quick Start:
        1. **Upload Dataset**: Upload your kidney ultrasound dataset as a zip file
        2. **Train Models**: Train AI models on your data
        3. **Predict**: Upload new images for classification
        4. **View Results**: Get instant classification with confidence scores

        ### 📁 Dataset Requirements:
        - Upload a zip file with this structure:
        ```
        dataset.zip
        ├── Normal/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── Stone/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
        ```
        - Supported formats: JPG, JPEG, PNG
        - Recommended: 100+ images per class

        ### 🏥 Medical Application:
        - Assist radiologists in kidney stone detection
        - Quick preliminary screening
        - Educational tool for medical students
        """)

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913503.png", width=200)

        # Check if dataset exists
        if os.path.exists("dataset"):
            st.success("✅ Dataset folder exists")
            # Count images
            normal_count = len(os.listdir("dataset/Normal")) if os.path.exists("dataset/Normal") else 0
            stone_count = len(os.listdir("dataset/Stone")) if os.path.exists("dataset/Stone") else 0
            st.metric("Normal Images", normal_count)
            st.metric("Stone Images", stone_count)
        else:
            st.warning("⚠️ No dataset found")


def upload_dataset_interface():
    st.header("📁 Upload Kidney Stone Dataset")

    st.markdown("""
    Upload your kidney ultrasound dataset as a zip file. The zip file should contain two folders:
    - **Normal/** - Images without kidney stones
    - **Stone/** - Images with kidney stones
    """)

    uploaded_file = st.file_uploader(
        "Choose a zip file containing your dataset",
        type=['zip'],
        help="Upload a zip file with 'Normal' and 'Stone' folders"
    )

    if uploaded_file is not None:
        st.info("📦 Uploading dataset...")

        # Remove existing dataset if it exists
        if os.path.exists("dataset"):
            shutil.rmtree("dataset")

        # Extract zip file
        if extract_zip_file(uploaded_file):
            # Verify dataset structure
            if os.path.exists("dataset/Normal") and os.path.exists("dataset/Stone"):
                normal_count = len(os.listdir("dataset/Normal"))
                stone_count = len(os.listdir("dataset/Stone"))

                st.success(f"✅ Dataset ready! Found {normal_count} Normal and {stone_count} Stone images.")

                # Show sample images
                st.subheader("Dataset Preview")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Normal Images**")
                    normal_files = os.listdir("dataset/Normal")[:3]
                    for file in normal_files:
                        st.write(f"📄 {file}")

                with col2:
                    st.write("**Stone Images**")
                    stone_files = os.listdir("dataset/Stone")[:3]
                    for file in stone_files:
                        st.write(f"📄 {file}")
            else:
                st.error("❌ Invalid dataset structure. Please include 'Normal' and 'Stone' folders.")
        else:
            st.error("❌ Failed to extract dataset.")


def train_models_interface(model_handler):
    st.header("🔄 Train Classification Models")

    # Check if dataset exists
    if not os.path.exists("dataset"):
        st.error("""
        ❌ No dataset found!

        Please go to the **'Upload Dataset'** section first to upload your kidney ultrasound dataset.
        """)
        return

    st.markdown("""
    Train three different neural network models for kidney stone classification.
    The models will be trained on your uploaded dataset.
    """)

    # Training parameters
    st.subheader("Training Configuration")
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("Number of Epochs", min_value=5, max_value=50, value=10)
    with col2:
        img_size = st.selectbox("Image Size", [64, 128], index=0, format_func=lambda x: f"{x}x{x}")
        model_handler.img_size = (img_size, img_size)
        model_handler.input_dim = img_size * img_size * 3

    # Model selection
    st.subheader("Model Selection")
    selected_models = st.multiselect(
        "Choose models to train:",
        ["DNN", "MLP", "AE-DNN"],
        default=["DNN", "MLP", "AE-DNN"]
    )

    if st.button("🚀 Start Training", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return

        with st.spinner("Loading dataset and training models... This may take several minutes."):
            try:
                # Load and preprocess data
                train_ds, val_ds, class_names = model_handler.load_and_preprocess_data()

                if train_ds is None:
                    return

                # Train models
                results = model_handler.train_models(train_ds, val_ds, epochs=epochs)

                st.success("🎉 Training completed successfully!")

                # Display results
                display_training_results(results)

                # Store in session state
                st.session_state.models_trained = True
                st.session_state.training_results = results

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")


def display_training_results(results):
    st.subheader("📊 Training Results")

    # Create metrics table
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            'Model': model_name,
            'Validation Accuracy': f"{result['val_accuracy']:.2%}",
            'Validation Loss': f"{result['val_loss']:.4f}",
            'Final Train Accuracy': f"{result['history']['accuracy'][-1]:.2%}",
            'Final Train Loss': f"{result['history']['loss'][-1]:.4f}"
        })

    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)

    # Create performance comparison chart
    st.subheader("Model Performance Comparison")

    fig = go.Figure()

    models = list(results.keys())
    val_accuracies = [results[model]['val_accuracy'] for model in models]

    fig.add_trace(go.Bar(
        x=models,
        y=val_accuracies,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        text=[f'{acc:.2%}' for acc in val_accuracies],
        textposition='auto',
    ))

    fig.update_layout(
        title="Validation Accuracy by Model",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Training history plots
    st.subheader("Training History")

    for model_name, result in results.items():
        col1, col2 = st.columns(2)

        with col1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                y=result['history']['accuracy'],
                name='Train Accuracy',
                line=dict(color='blue')
            ))
            fig_acc.add_trace(go.Scatter(
                y=result['history']['val_accuracy'],
                name='Val Accuracy',
                line=dict(color='red', dash='dash')
            ))
            fig_acc.update_layout(
                title=f'{model_name} - Accuracy',
                height=300
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=result['history']['loss'],
                name='Train Loss',
                line=dict(color='blue')
            ))
            fig_loss.add_trace(go.Scatter(
                y=result['history']['val_loss'],
                name='Val Loss',
                line=dict(color='red', dash='dash')
            ))
            fig_loss.update_layout(
                title=f'{model_name} - Loss',
                height=300
            )
            st.plotly_chart(fig_loss, use_container_width=True)


def predict_interface(model_handler):
    st.header("🔍 Predict Kidney Stone")

    # Check if models are trained
    if not hasattr(model_handler, 'models') or not model_handler.models:
        st.warning("""
        ⚠️ **No trained models found!**

        Please go to the **'Train Models'** section first to train the classification models.
        """)
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a kidney ultrasound image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a kidney ultrasound image for classification"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Model selection
            st.subheader("Model Selection")
            selected_model = st.selectbox(
                "Choose model for prediction:",
                list(model_handler.models.keys()),
                help="Select which trained model to use for prediction"
            )

            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Make prediction
                        prediction, confidence, probabilities = model_handler.predict_image(
                            image, selected_model
                        )

                        # Display results in the right column
                        with col2:
                            display_prediction_results(
                                prediction, confidence, probabilities,
                                selected_model, image
                            )

                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")


def display_prediction_results(prediction, confidence, probabilities, model_name, image):
    st.subheader("Analysis Results")

    # Prediction result with emoji and color
    if prediction == "Stone":
        emoji = "💎"
        color = "red"
        box_class = "stone"
        advice = """
        **Medical Advice:**
        - Consult a urologist for further evaluation
        - Drink plenty of water
        - Consider follow-up imaging studies
        - Monitor for symptoms like pain or blood in urine
        """
    else:
        emoji = "✅"
        color = "green"
        box_class = "normal"
        advice = """
        **Medical Advice:**
        - No kidney stones detected in this image
        - Maintain healthy hydration
        - Regular check-ups recommended
        - Continue healthy lifestyle habits
        """

    # Prediction box
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h2>{emoji} Prediction: {prediction}</h2>
        <h3>Confidence: {confidence:.2%}</h3>
        <p>Model: {model_name}</p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence metrics
    st.subheader("Confidence Analysis")

    # Probability chart
    fig = go.Figure(go.Bar(
        x=list(probabilities.keys()),
        y=list(probabilities.values()),
        marker_color=['green', 'red'],
        text=[f'{prob:.2%}' for prob in probabilities.values()],
        textposition='auto',
    ))

    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model information
    st.subheader("Model Information")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Selected Model", model_name)
        st.metric("Prediction", f"{emoji} {prediction}")

    with col2:
        st.metric("Confidence Score", f"{confidence:.2%}")
        st.metric("Image Size", f"{image.size[0]}x{image.size[1]}")

    # Medical advice
    st.subheader("Medical Guidance")
    if prediction == "Stone":
        st.warning(advice)
    else:
        st.success(advice)

    # Disclaimer
    st.info("""
    **Disclaimer:** This is a demonstration tool. For actual medical diagnosis, 
    always consult with qualified healthcare professionals and rely on comprehensive clinical evaluation.
    """)


def show_model_info(model_handler):
    st.header("📊 Model Information")

    st.markdown("""
    ## Model Architectures

    ### 1. DNN (Deep Neural Network)
    - **Layers**: 3 hidden layers (512, 256 neurons)
    - **Regularization**: L2 regularization + Dropout
    - **Features**: Batch normalization, ReLU activation
    - **Use Case**: High-capacity model for complex patterns

    ### 2. MLP (Multi-Layer Perceptron)
    - **Layers**: 2 hidden layers (128, 64 neurons)
    - **Regularization**: L2 regularization + Dropout
    - **Features**: Lightweight, fast training
    - **Use Case**: Efficient model for quick predictions

    ### 3. AE-DNN (Autoencoder + DNN)
    - **Architecture**: Autoencoder for feature extraction + Classifier
    - **Features**: Learns compressed representations
    - **Use Case**: Feature learning + classification

    ## Technical Specifications
    - **Input Size**: 64x64 RGB images (12,288 features)
    - **Output**: Binary classification (Normal/Stone)
    - **Activation**: Softmax for multi-class probability
    - **Optimizer**: Adam optimizer
    - **Loss Function**: Sparse Categorical Crossentropy
    """)

    # Show trained models status
    if hasattr(model_handler, 'models') and model_handler.models:
        st.subheader("Trained Models")
        for model_name in model_handler.models.keys():
            st.success(f"✅ {model_name} - Ready for predictions")
    else:
        st.warning("⚠️ No models trained yet")


if __name__ == "__main__":
    main()