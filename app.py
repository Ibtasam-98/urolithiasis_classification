# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras import models, layers, regularizers

# Set page configuration
st.set_page_config(
    page_title="Kidney Stone Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .live-badge {
        background-color: #28a745;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .prediction-card {
        background-color: white;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
class Config:
    SEED = 42
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 16  # Reduced batch size for memory

config = Config()

# Model names
MODEL_NAMES = {
    "DNN": "Deep Neural Network",
    "MLP": "Multi-Layer Perceptron", 
    "AE-DNN": "Autoencoder DNN"
}

def set_seeds():
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

def create_dnn_model(input_dim):
    """Simplified DNN model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"), 
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_mlp_model(input_dim):
    """Simplified MLP model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_autoencoder_dnn_model(input_dim):
    """Simplified Autoencoder model"""
    # Encoder
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation="relu")(inputs)
    encoded = layers.Dense(64, activation="relu")(encoded)
    
    # Classifier
    outputs = layers.Dense(2, activation="softmax")(encoded)
    
    classifier = tf.keras.Model(inputs, outputs)
    classifier.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return classifier

def load_sample_data():
    """Create sample data for demonstration"""
    try:
        # Create dummy dataset structure
        if not os.path.exists("dataset"):
            os.makedirs("dataset/stone", exist_ok=True)
            os.makedirs("dataset/normal", exist_ok=True)
            st.info("Sample dataset structure created. Please add your kidney stone images to dataset/stone/ and normal images to dataset/normal/")
            return None, None, None, ['normal', 'stone']
        
        # Try to load actual data
        dataset = tf.keras.utils.image_dataset_from_directory(
            "dataset",
            image_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            validation_split=0.2,
            subset="training",
            seed=config.SEED
        )
        class_names = dataset.class_names
        
        # Create train/val split
        train_size = int(0.8 * len(dataset))
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)
        
        return train_ds, val_ds, dataset, class_names
        
    except Exception as e:
        st.warning(f"Could not load dataset: {str(e)}")
        st.info("Please make sure you have a 'dataset' folder with 'stone' and 'normal' subfolders containing images.")
        return None, None, None, ['normal', 'stone']

def preprocess_datasets(train_ds, val_ds, test_ds):
    """Normalize and flatten datasets"""
    try:
        normalization = tf.keras.layers.Rescaling(1./255)
        
        def preprocess(ds):
            return ds.map(lambda x, y: (tf.reshape(normalization(x), (-1, config.IMG_SIZE[0] * config.IMG_SIZE[1] * 3)), y))
        
        train_flat = preprocess(train_ds)
        val_flat = preprocess(val_ds) 
        test_flat = preprocess(test_ds)
        
        input_dim = config.IMG_SIZE[0] * config.IMG_SIZE[1] * 3
        return train_flat, val_flat, test_flat, input_dim
        
    except Exception as e:
        st.error(f"Error preprocessing: {str(e)}")
        return None, None, None, 0

def evaluate_simple_model(model, train_ds, val_ds, test_ds, model_name):
    """Simplified evaluation"""
    try:
        # Basic metrics
        train_loss, train_acc = model.evaluate(train_ds, verbose=0)
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        
        return {
            "train_acc": train_acc,
            "val_acc": val_acc, 
            "test_acc": test_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "metrics": {
                'accuracy': test_acc,
                'precision': 0.8,  # Placeholder
                'recall': 0.8,     # Placeholder
                'f1_score': 0.8,   # Placeholder
                'roc_auc': 0.8     # Placeholder
            }
        }
    except Exception as e:
        st.error(f"Evaluation error for {model_name}: {str(e)}")
        return None

def plot_training_history(history, model_name):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val') 
    ax2.set_title(f'{model_name} - Loss')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def preprocess_image(image):
    """Preprocess uploaded image"""
    image = image.resize(config.IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_flat = img_array.reshape(1, -1)
    return img_flat

def main():
    # Header with Live badge
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">Kidney Stone Detection</h1>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="live-badge">🟢 LIVE</div>', unsafe_allow_html=True)
    
    st.markdown("""
    AI-powered kidney stone detection using deep learning.
    Compare multiple neural network models with performance analysis.
    """)

    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Train Models", "Predict"])

    with tab1:
        st.header("Welcome")
        st.markdown("""
        This app demonstrates kidney stone detection using deep learning.
        
        **Models:**
        - Deep Neural Network (DNN)
        - Multi-Layer Perceptron (MLP) 
        - Autoencoder-based DNN
        
        **How to use:**
        1. Add kidney images to dataset/stone/ and normal images to dataset/normal/
        2. Go to Train Models tab and click 'Start Training'
        3. Use Predict tab to classify new images
        """)

    with tab2:
        st.header("Model Training")
        
        if not st.session_state.trained:
            st.subheader("Training Setup")
            epochs = st.slider("Epochs", 2, 10, 3)
            
            if st.button("Start Training", type="primary"):
                with st.spinner("Loading data..."):
                    set_seeds()
                    train_ds, val_ds, test_ds, class_names = load_sample_data()
                    
                    if train_ds is None:
                        st.error("Please add images to the dataset folder first!")
                        return
                    
                    # Preprocess
                    train_flat, val_flat, test_flat, input_dim = preprocess_datasets(train_ds, val_ds, test_ds)
                    
                    if train_flat is None:
                        st.error("Data preprocessing failed!")
                        return

                # Train models
                models_to_train = ["DNN", "MLP", "AE-DNN"]
                progress_bar = st.progress(0)
                
                for i, model_name in enumerate(models_to_train):
                    st.write(f"Training {MODEL_NAMES[model_name]}...")
                    
                    try:
                        if model_name == "DNN":
                            model = create_dnn_model(input_dim)
                        elif model_name == "MLP":
                            model = create_mlp_model(input_dim) 
                        else:
                            model = create_autoencoder_dnn_model(input_dim)
                        
                        # Train with progress
                        history = model.fit(
                            train_flat, 
                            validation_data=val_flat,
                            epochs=epochs,
                            verbose=0
                        )
                        
                        # Evaluate
                        results = evaluate_simple_model(model, train_flat, val_flat, test_flat, model_name)
                        
                        if results:
                            st.session_state.models[model_name] = model
                            st.session_state.results[model_name] = {
                                **results,
                                'history': history
                            }
                            st.success(f"✓ {MODEL_NAMES[model_name]} trained!")
                        
                    except Exception as e:
                        st.error(f"Failed to train {model_name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(models_to_train))
                    time.sleep(0.5)
                
                if st.session_state.models:
                    st.session_state.trained = True
                    st.balloons()
                    st.success("All models trained successfully!")
                    st.rerun()
                else:
                    st.error("Training failed for all models!")

        else:
            # Show results
            st.subheader("Training Results")
            
            # Performance table
            st.write("**Model Performance:**")
            perf_data = []
            for name, results in st.session_state.results.items():
                perf_data.append({
                    'Model': MODEL_NAMES[name],
                    'Train Acc': f"{results['train_acc']:.3f}",
                    'Val Acc': f"{results['val_acc']:.3f}", 
                    'Test Acc': f"{results['test_acc']:.3f}"
                })
            
            st.dataframe(pd.DataFrame(perf_data))
            
            # Training plots
            st.write("**Training History:**")
            for name, results in st.session_state.results.items():
                with st.expander(f"{MODEL_NAMES[name]} Training"):
                    fig = plot_training_history(results['history'], MODEL_NAMES[name])
                    st.pyplot(fig)
            
            if st.button("Retrain Models"):
                st.session_state.trained = False
                st.session_state.models = {}
                st.session_state.results = {}
                st.rerun()

    with tab3:
        st.header("Image Prediction")
        
        if not st.session_state.trained:
            st.info("Please train models first!")
        else:
            uploaded = st.file_uploader("Upload kidney ultrasound image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded:
                img = Image.open(uploaded)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Classify Image"):
                    processed = preprocess_image(img)
                    predictions = {}
                    
                    for name, model in st.session_state.models.items():
                        pred = model.predict(processed, verbose=0)
                        stone_prob = pred[0][1]
                        normal_prob = pred[0][0]
                        
                        predictions[MODEL_NAMES[name]] = {
                            'prediction': 'Stone' if stone_prob > 0.5 else 'Normal',
                            'confidence': max(stone_prob, normal_prob),
                            'stone_prob': stone_prob,
                            'normal_prob': normal_prob
                        }
                    
                    with col2:
                        st.write("**Predictions:**")
                        for model_name, pred in predictions.items():
                            color = "#dc3545" if pred['prediction'] == 'Stone' else "#28a745"
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3 style="color: {color}; margin: 0;">{model_name}</h3>
                                <p><strong>Result:</strong> {pred['prediction']}</p>
                                <p><strong>Confidence:</strong> {pred['confidence']:.3f}</p>
                                <p>Stone: {pred['stone_prob']:.3f} | Normal: {pred['normal_prob']:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Consensus
                    stone_votes = sum(1 for p in predictions.values() if p['prediction'] == 'Stone')
                    total = len(predictions)
                    consensus = "Stone" if stone_votes > total/2 else "Normal"
                    
                    st.success(f"**Final Consensus: {consensus}** ({stone_votes}/{total} models agree)")

if __name__ == "__main__":
    main()
