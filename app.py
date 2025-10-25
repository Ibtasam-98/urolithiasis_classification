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
import io
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, jaccard_score, fowlkes_mallows_score
)
from tensorflow.keras import models, layers, regularizers

# Set page configuration
st.set_page_config(
    page_title="Kidney Stone Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .warning-text {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-text {
        color: #dc3545;
        font-weight: bold;
    }
    .prediction-card {
        background-color: white;
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .live-badge {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class Config:
    SEED = 123
    DATA_DIR = "dataset"
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32


config = Config()

# Model names mapping
MODEL_NAMES = {
    "DNN": "Deep Neural Network",
    "MLP": "Multi-Layer Perceptron",
    "AE-DNN": "Autoencoder-based Deep Neural Network"
}

def set_seeds():
    """Set all random seeds for reproducibility"""
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)


def load_datasets():
    """Load and split the dataset into train, validation, and test sets"""
    try:
        # Load full dataset to get class names
        full_ds = tf.keras.utils.image_dataset_from_directory(
            config.DATA_DIR,
            seed=config.SEED,
            image_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        class_names = full_ds.class_names

        # Split into train+val (80%) and test (20%)
        train_val_ds = tf.keras.utils.image_dataset_from_directory(
            config.DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=config.SEED,
            image_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            config.DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=config.SEED,
            image_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=False  # No shuffle for test set
        )

        # Split train_val into train (80%) and val (20%)
        train_size = int(0.8 * len(list(train_val_ds)))
        train_ds = train_val_ds.take(train_size)
        val_ds = train_val_ds.skip(train_size)

        # Cache/prefetch for performance with better error handling
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Use try-except for dataset operations
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, test_ds, class_names
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None, None, None, []


def preprocess_datasets(train_ds, val_ds, test_ds):
    """Normalize and flatten datasets with error handling"""
    try:
        norm = layers.Rescaling(1. / 255)

        def map_norm_flat(ds):
            return ds.map(lambda x, y: (tf.reshape(norm(x), [tf.shape(x)[0], -1]), y), 
                         num_parallel_calls=tf.data.AUTOTUNE)

        train_ds_flat = map_norm_flat(train_ds)
        val_ds_flat = map_norm_flat(val_ds)
        test_ds_flat = map_norm_flat(test_ds)

        INPUT_DIM = config.IMG_SIZE[0] * config.IMG_SIZE[1] * 3

        return train_ds_flat, val_ds_flat, test_ds_flat, INPUT_DIM
    except Exception as e:
        st.error(f"Error preprocessing datasets: {str(e)}")
        return None, None, None, 0


def create_dnn_model(input_dim):
    """Create Deep Neural Network model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def create_mlp_model(input_dim):
    """Create smaller MLP model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def create_autoencoder_dnn_model(input_dim):
    """Create Autoencoder-based DNN classifier"""
    # Autoencoder
    inputs = tf.keras.Input(shape=(input_dim,))
    # Encoder
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    latent = layers.Dense(128, activation="relu")(x)
    # Decoder
    y = layers.Dense(256, activation="relu")(latent)
    y = layers.Dense(512, activation="relu")(y)
    decoded = layers.Dense(input_dim, activation="sigmoid")(y)

    # Autoencoder model
    autoencoder = tf.keras.Model(inputs, decoded, name="autoencoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    # Classifier
    classifier_output = layers.Dense(2, activation="softmax")(latent)
    classifier = tf.keras.Model(inputs, classifier_output, name="autoencoder_classifier")
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return autoencoder, classifier


def calculate_all_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive evaluation metrics"""
    try:
        # Basic metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # NPV (Negative Predictive Value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # ROC AUC
        fpr_roc, tpr_roc, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr_roc, tpr_roc)

        # PR AUC
        precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall_pr, precision_pr)

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}


def evaluate_model(model, train_ds, val_ds, test_ds, model_name, class_names, history=None):
    """Evaluate a model and return comprehensive metrics"""
    try:
        # Evaluate metrics
        train_loss, train_acc = model.evaluate(train_ds, verbose=0)
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)

        # Get predictions - convert dataset to numpy arrays first
        stone_idx = class_names.index('stone') if 'stone' in class_names else 1
        
        # Convert validation dataset to numpy arrays
        val_images, val_labels = [], []
        for x, y in val_ds:
            val_images.append(x.numpy())
            val_labels.append(y.numpy())
        
        val_images = np.concatenate(val_images)
        val_labels = np.concatenate(val_labels)
        
        # Get predictions
        y_probs = model.predict(val_images, verbose=0)[:, stone_idx]
        y_pred = (y_probs >= 0.5).astype(int)

        # Calculate all metrics
        metrics = calculate_all_metrics(val_labels, y_pred, y_probs)

        return {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "metrics": metrics,
            "history": history.history if history else None,
            "y_true": val_labels,
            "y_probs": y_probs
        }
    except Exception as e:
        st.error(f"Error evaluating model {model_name}: {str(e)}")
        return None


def plot_training_history(history, model_name):
    """Plot training history for a single model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
    ax1.set_title(f'{MODEL_NAMES.get(model_name, model_name)} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history['loss'], label='Training Loss', linewidth=2, color='blue')
    ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    ax2.set_title(f'{MODEL_NAMES.get(model_name, model_name)} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_metrics(results_dict):
    """Create comparison visualizations"""
    model_names = list(results_dict.keys())

    # Key metrics for comparison
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Bar chart for key metrics
    metrics_values = {metric: [results_dict[model]['metrics'][metric] for model in model_names]
                      for metric in key_metrics}

    x = np.arange(len(model_names))
    width = 0.15
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']

    for i, metric in enumerate(key_metrics):
        axes[0].bar(x + i * width, metrics_values[metric], width, label=metric, color=colors[i])

    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Key Metrics Comparison')
    axes[0].set_xticks(x + width * 2)
    axes[0].set_xticklabels([MODEL_NAMES.get(model, model) for model in model_names])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ROC Curves
    for i, model_name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(results_dict[model_name]['y_true'],
                                results_dict[model_name]['y_probs'])
        roc_auc = results_dict[model_name]['metrics']['roc_auc']
        axes[2].plot(fpr, tpr, label=f'{MODEL_NAMES.get(model_name, model_name)} (AUC = {roc_auc:.3f})', linewidth=2)

    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC Curves')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Accuracy comparison
    acc_types = ['train_acc', 'val_acc', 'test_acc']
    acc_labels = ['Training', 'Validation', 'Test']
    acc_values = {acc_type: [results_dict[model][acc_type] for model in model_names]
                  for acc_type in acc_types}

    for i, acc_type in enumerate(acc_types):
        axes[4].bar(x + i * width, acc_values[acc_type], width, label=acc_labels[i], color=colors[i])

    axes[4].set_xlabel('Models')
    axes[4].set_ylabel('Accuracy')
    axes[4].set_title('Accuracy Comparison')
    axes[4].set_xticks(x + width)
    axes[4].set_xticklabels([MODEL_NAMES.get(model, model) for model in model_names])
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    # Hide unused subplots
    axes[1].set_visible(False)
    axes[3].set_visible(False)
    axes[5].set_visible(False)

    plt.tight_layout()
    return fig


def preprocess_image(image, img_size=(64, 64)):
    """Preprocess uploaded image for prediction"""
    # Resize image
    image = image.resize(img_size)
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    # Flatten the image
    img_array_flat = img_array.reshape(1, -1)
    return img_array_flat


def main():
    # Header with Live status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">Kidney Stone Detection System</h1>', 
                    unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="live-badge">
            🟢 LIVE
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    AI-powered kidney stone detection using deep learning.
    Compare multiple neural network models with comprehensive performance analysis.
    """)

    # Initialize session state for models and results
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'results_dict' not in st.session_state:
        st.session_state.results_dict = {}
    if 'class_names' not in st.session_state:
        st.session_state.class_names = []
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}

    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Train Models", "Predict"])

    with tab1:
        st.header("Welcome to Kidney Stone Detection System")
        st.markdown("""
        ### Overview
        This application provides a comprehensive platform for training, evaluating, and comparing 
        different deep learning models for kidney stone detection from ultrasound images.

        ### Features:
        - **Train Models**: Train three different neural network architectures
        - **View Results**: Compare model performance with detailed metrics and visualizations
        - **Make Predictions**: Upload kidney ultrasound images for classification

        ### Model Architectures:
        1. **Deep Neural Network (DNN)**: Large network with 512-256 neurons
        2. **Multi-Layer Perceptron (MLP)**: Smaller network with 128-64 neurons
        3. **Autoencoder-based Deep Neural Network (AE-DNN)**: Autoencoder-based feature extraction + classification

        ### How to Use:
        1. Go to the **Train Models** tab to train all models
        2. View the results and visualizations in the same tab after training
        3. Use the **Predict** tab to classify new ultrasound images
        """)

    with tab2:
        st.header("Model Training and Results")

        if not st.session_state.models_trained:
            # Training section
            st.subheader("Training Configuration")
            epochs = st.slider("Number of Epochs", min_value=3, max_value=15, value=5)

            st.subheader("Models to be Trained")
            st.markdown("""
            The following models will be trained:
            - **Deep Neural Network (DNN)**
            - **Multi-Layer Perceptron (MLP)**
            - **Autoencoder-based Deep Neural Network (AE-DNN)**
            """)

            if st.button("Start Training All Models", type="primary"):
                models_to_train = ["DNN", "MLP", "AE-DNN"]

                with st.spinner("Loading and preprocessing data..."):
                    set_seeds()
                    train_ds, val_ds, test_ds, class_names = load_datasets()

                    if train_ds is None:
                        st.error("Failed to load dataset. Please check your dataset folder structure.")
                    else:
                        st.session_state.train_ds, st.session_state.val_ds, st.session_state.test_ds = train_ds, val_ds, test_ds
                        st.session_state.class_names = class_names

                        train_ds_flat, val_ds_flat, test_ds_flat, INPUT_DIM = preprocess_datasets(train_ds, val_ds, test_ds)
                        
                        if train_ds_flat is None:
                            st.error("Error preprocessing datasets.")
                        else:
                            st.session_state.train_ds_flat = train_ds_flat
                            st.session_state.val_ds_flat = val_ds_flat
                            st.session_state.test_ds_flat = test_ds_flat
                            st.session_state.INPUT_DIM = INPUT_DIM

                            st.success(f"Data loaded successfully! Found {len(class_names)} classes: {class_names}")

                # Training progress
                st.session_state.results_dict = {}
                st.session_state.trained_models = {}

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, model_name in enumerate(models_to_train):
                    full_model_name = MODEL_NAMES.get(model_name, model_name)
                    status_text.text(f"Training {full_model_name}... ({i + 1}/{len(models_to_train)})")

                    try:
                        if model_name == "DNN":
                            model = create_dnn_model(st.session_state.INPUT_DIM)
                            history = model.fit(
                                st.session_state.train_ds_flat,
                                validation_data=st.session_state.val_ds_flat,
                                epochs=epochs,
                                verbose=0
                            )
                            results = evaluate_model(
                                model,
                                st.session_state.train_ds_flat,
                                st.session_state.val_ds_flat,
                                st.session_state.test_ds_flat,
                                model_name,
                                st.session_state.class_names,
                                history
                            )
                            if results:
                                st.session_state.trained_models["DNN"] = model
                                st.session_state.results_dict["DNN"] = results
                                st.success(f"{full_model_name} trained successfully!")

                        elif model_name == "MLP":
                            model = create_mlp_model(st.session_state.INPUT_DIM)
                            history = model.fit(
                                st.session_state.train_ds_flat,
                                validation_data=st.session_state.val_ds_flat,
                                epochs=epochs,
                                verbose=0
                            )
                            results = evaluate_model(
                                model,
                                st.session_state.train_ds_flat,
                                st.session_state.val_ds_flat,
                                st.session_state.test_ds_flat,
                                model_name,
                                st.session_state.class_names,
                                history
                            )
                            if results:
                                st.session_state.trained_models["MLP"] = model
                                st.session_state.results_dict["MLP"] = results
                                st.success(f"{full_model_name} trained successfully!")

                        elif model_name == "AE-DNN":
                            autoencoder, classifier = create_autoencoder_dnn_model(st.session_state.INPUT_DIM)
                            # Train autoencoder
                            autoencoder.fit(
                                st.session_state.train_ds_flat.map(lambda x, y: (x, x)),
                                validation_data=st.session_state.val_ds_flat.map(lambda x, y: (x, x)),
                                epochs=epochs,
                                verbose=0
                            )
                            # Train classifier
                            history = classifier.fit(
                                st.session_state.train_ds_flat,
                                validation_data=st.session_state.val_ds_flat,
                                epochs=epochs,
                                verbose=0
                            )
                            results = evaluate_model(
                                classifier,
                                st.session_state.train_ds_flat,
                                st.session_state.val_ds_flat,
                                st.session_state.test_ds_flat,
                                model_name,
                                st.session_state.class_names,
                                history
                            )
                            if results:
                                st.session_state.trained_models["AE-DNN"] = classifier
                                st.session_state.results_dict["AE-DNN"] = results
                                st.success(f"{full_model_name} trained successfully!")

                    except Exception as e:
                        st.error(f"Error training {full_model_name}: {str(e)}")

                    progress_bar.progress((i + 1) / len(models_to_train))
                    time.sleep(1)  # Visual feedback

                if st.session_state.results_dict:
                    st.session_state.models_trained = True
                    st.success("All models trained successfully! View the results below.")
                    st.rerun()
                else:
                    st.error("Training failed for all models. Please check the dataset and try again.")

        else:
            # Display results if models are trained
            if st.session_state.results_dict:
                st.subheader("Training Results")

                # Overall comparison table
                st.subheader("Overall Performance Comparison")
                comparison_data = []
                for model_name, results in st.session_state.results_dict.items():
                    full_model_name = MODEL_NAMES.get(model_name, model_name)
                    row = {
                        'Model': full_model_name,
                        'Train Acc': f"{results['train_acc']:.3f}",
                        'Val Acc': f"{results['val_acc']:.3f}",
                        'Test Acc': f"{results['test_acc']:.3f}",
                        'Precision': f"{results['metrics']['precision']:.3f}",
                        'Recall': f"{results['metrics']['recall']:.3f}",
                        'F1-Score': f"{results['metrics']['f1_score']:.3f}",
                        'ROC AUC': f"{results['metrics']['roc_auc']:.3f}"
                    }
                    comparison_data.append(row)

                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)

                # Individual model results
                st.subheader("Individual Model Performance")
                for model_name, results in st.session_state.results_dict.items():
                    full_model_name = MODEL_NAMES.get(model_name, model_name)
                    with st.expander(f"{full_model_name} Details", expanded=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Training Accuracy", f"{results['train_acc']:.3f}")
                            st.metric("Training Loss", f"{results['train_loss']:.3f}")

                        with col2:
                            st.metric("Validation Accuracy", f"{results['val_acc']:.3f}")
                            st.metric("Validation Loss", f"{results['val_loss']:.3f}")

                        with col3:
                            st.metric("Test Accuracy", f"{results['test_acc']:.3f}")
                            st.metric("Test Loss", f"{results['test_loss']:.3f}")

                        # Plot training history
                        if results['history']:
                            fig = plot_training_history(results['history'], model_name)
                            st.pyplot(fig)

                # Visual comparisons
                st.subheader("Model Comparison Visualizations")
                fig = plot_comparison_metrics(st.session_state.results_dict)
                st.pyplot(fig)

                # Best model recommendation
                best_model = max(st.session_state.results_dict.keys(),
                                 key=lambda x: st.session_state.results_dict[x]['test_acc'])
                best_accuracy = st.session_state.results_dict[best_model]['test_acc']
                best_model_full_name = MODEL_NAMES.get(best_model, best_model)

                st.success(f"""
                **Best Performing Model**: **{best_model_full_name}** 
                With Test Accuracy: **{best_accuracy:.3f}**
                """)

            else:
                st.error("No training results available. Please retrain the models.")

        # Button to retrain models
        if st.session_state.models_trained:
            if st.button("Retrain Models"):
                st.session_state.models_trained = False
                st.session_state.results_dict = {}
                st.session_state.trained_models = {}
                st.rerun()

    with tab3:
        st.header("Image Prediction")

        if not st.session_state.models_trained:
            st.info("Please train models first in the 'Train Models' tab to enable predictions.")
        else:
            st.subheader("Upload Kidney Ultrasound Image")
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Predict", type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Preprocess image
                        processed_image = preprocess_image(image)

                        predictions = {}

                        for model_name, model in st.session_state.trained_models.items():
                            # Get prediction
                            prediction = model.predict(processed_image, verbose=0)
                            stone_prob = prediction[0][1]  # Probability of stone class
                            normal_prob = prediction[0][0]  # Probability of normal class

                            full_model_name = MODEL_NAMES.get(model_name, model_name)
                            predictions[full_model_name] = {
                                'stone_probability': stone_prob,
                                'normal_probability': normal_prob,
                                'prediction': 'Stone' if stone_prob > 0.5 else 'Normal',
                                'confidence': max(stone_prob, normal_prob)
                            }

                        # Display predictions
                        with col2:
                            st.subheader("Prediction Results")

                            for model_name, pred in predictions.items():
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h3 style="color: #dc3545; margin-top: 0;">{model_name}</h3>
                                    <p><strong>Prediction:</strong> {pred['prediction']}</p>
                                    <p><strong>Confidence:</strong> {pred['confidence']:.3f}</p>
                                    <p><strong>Stone Probability:</strong> {pred['stone_probability']:.3f}</p>
                                    <p><strong>Normal Probability:</strong> {pred['normal_probability']:.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)

                        # Consensus prediction
                        stone_votes = sum(1 for pred in predictions.values() if pred['prediction'] == 'Stone')
                        total_votes = len(predictions)

                        st.subheader("Consensus Analysis")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Stone Votes", stone_votes)
                        with col2:
                            st.metric("Normal Votes", total_votes - stone_votes)
                        with col3:
                            consensus = "Stone" if stone_votes > total_votes / 2 else "Normal"
                            consensus_color = "#dc3545" if consensus == "Stone" else "#28a745"
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <h3 style="color: {consensus_color};">Final Consensus</h3>
                                <h2 style="color: {consensus_color};">{consensus}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        # Confidence comparison
                        st.subheader("Model Confidence Levels")
                        conf_data = {
                            'Model': list(predictions.keys()),
                            'Confidence': [pred['confidence'] for pred in predictions.values()],
                            'Prediction': [pred['prediction'] for pred in predictions.values()]
                        }
                        conf_df = pd.DataFrame(conf_data)
                        
                        # Style the dataframe with red for Stone predictions
                        def style_predictions(val):
                            color = '#dc3545' if val == 'Stone' else '#28a745'
                            return f'color: {color}; font-weight: bold;'
                        
                        styled_df = conf_df.style.applymap(style_predictions, subset=['Prediction'])
                        st.dataframe(styled_df, use_container_width=True)


if __name__ == "__main__":
    main()
