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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
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
    .metric-red {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-white {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)


# Configuration
class Config:
    SEED = 42
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 16


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
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation="relu")(inputs)
    encoded = layers.Dense(64, activation="relu")(encoded)
    outputs = layers.Dense(2, activation="softmax")(encoded)
    classifier = tf.keras.Model(inputs, outputs)
    classifier.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return classifier


def load_sample_data():
    try:
        if not os.path.exists("dataset"):
            os.makedirs("dataset/stone", exist_ok=True)
            os.makedirs("dataset/normal", exist_ok=True)
            st.info(
                "Sample dataset structure created. Please add your kidney stone images to dataset/stone/ and normal images to dataset/normal/")
            return None, None, None, ['normal', 'stone']

        dataset = tf.keras.utils.image_dataset_from_directory(
            "dataset",
            image_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            validation_split=0.2,
            subset="training",
            seed=config.SEED
        )
        class_names = dataset.class_names

        train_size = int(0.8 * len(dataset))
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)

        return train_ds, val_ds, dataset, class_names

    except Exception as e:
        st.warning(f"Could not load dataset: {str(e)}")
        st.info("Please make sure you have a 'dataset' folder with 'stone' and 'normal' subfolders containing images.")
        return None, None, None, ['normal', 'stone']


def preprocess_datasets(train_ds, val_ds, test_ds):
    try:
        normalization = tf.keras.layers.Rescaling(1. / 255)

        def preprocess(ds):
            return ds.map(
                lambda x, y: (tf.reshape(normalization(x), (-1, config.IMG_SIZE[0] * config.IMG_SIZE[1] * 3)), y))

        train_flat = preprocess(train_ds)
        val_flat = preprocess(val_ds)
        test_flat = preprocess(test_ds)

        input_dim = config.IMG_SIZE[0] * config.IMG_SIZE[1] * 3
        return train_flat, val_flat, test_flat, input_dim

    except Exception as e:
        st.error(f"Error preprocessing: {str(e)}")
        return None, None, None, 0


def get_detailed_predictions(model, dataset):
    """Get detailed predictions for evaluation"""
    y_true = []
    y_pred = []
    y_probs = []

    for x, y in dataset:
        predictions = model.predict(x, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_probs.extend(predictions[:, 1])  # Probability of class 1 (stone)

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def evaluate_model_comprehensive(model, train_ds, val_ds, test_ds, model_name, class_names):
    """Comprehensive model evaluation with detailed metrics"""
    try:
        # Basic metrics
        train_loss, train_acc = model.evaluate(train_ds, verbose=0)
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)

        # Get detailed predictions for test set
        y_true, y_pred, y_probs = get_detailed_predictions(model, test_ds)

        # Calculate comprehensive metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        return {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "metrics": {
                'accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            "confusion_matrix": cm,
            "classification_report": class_report,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs
        }
    except Exception as e:
        st.error(f"Evaluation error for {model_name}: {str(e)}")
        return None


def plot_training_history(history, model_name):
    """Plot training progress with enhanced visuals"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
    ax1.set_title(f'{model_name} - Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    ax2.set_title(f'{model_name} - Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning dynamics - Accuracy difference
    acc_diff = [val - train for train, val in zip(history.history['accuracy'], history.history['val_accuracy'])]
    ax3.plot(acc_diff, label='Val - Train Accuracy', linewidth=2, color='green')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_title(f'{model_name} - Learning Dynamics', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Convergence speed
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax4.plot(epochs, history.history['accuracy'], label='Training', linewidth=2)
    ax4.plot(epochs, history.history['val_accuracy'], label='Validation', linewidth=2)
    ax4.set_title(f'{model_name} - Convergence Analysis', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig


def plot_metrics_comparison(results_dict):
    """Plot comparison of all models' metrics"""
    model_names = [MODEL_NAMES[name] for name in results_dict.keys()]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    # Bar chart for all metrics
    for i, metric in enumerate(metrics):
        values = [results_dict[model]['metrics'][metric] for model in results_dict.keys()]
        axes[i].bar(model_names, values, color=colors[:len(model_names)])
        axes[i].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # Training vs Validation accuracy comparison
    train_acc = [results_dict[model]['train_acc'] for model in results_dict.keys()]
    val_acc = [results_dict[model]['val_acc'] for model in results_dict.keys()]

    x = np.arange(len(model_names))
    width = 0.35

    axes[4].bar(x - width / 2, train_acc, width, label='Training', color='blue', alpha=0.7)
    axes[4].bar(x + width / 2, val_acc, width, label='Validation', color='red', alpha=0.7)
    axes[4].set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    axes[4].set_ylabel('Accuracy')
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(model_names, rotation=45)
    axes[4].legend()

    # Hide the last subplot
    axes[5].set_visible(False)

    plt.tight_layout()
    return fig


def plot_roc_curves(results_dict, class_names):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (model_key, results) in enumerate(results_dict.items()):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_probs'])
        roc_auc = results['metrics']['roc_auc']

        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{MODEL_NAMES[model_key]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig


def display_comprehensive_results(results_dict, class_names):
    """Display all comprehensive results and visualizations"""

    st.header("📊 Comprehensive Model Analysis")

    # 1. Performance Summary Table
    st.subheader("📋 Performance Summary")
    summary_data = []
    for model_name, results in results_dict.items():
        summary_data.append({
            'Model': MODEL_NAMES[model_name],
            'Train Accuracy': f"{results['train_acc']:.4f}",
            'Val Accuracy': f"{results['val_acc']:.4f}",
            'Test Accuracy': f"{results['test_acc']:.4f}",
            'Precision': f"{results['metrics']['precision']:.4f}",
            'Recall': f"{results['metrics']['recall']:.4f}",
            'F1-Score': f"{results['metrics']['f1_score']:.4f}",
            'ROC AUC': f"{results['metrics']['roc_auc']:.4f}"
        })

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # 2. Metrics Comparison Visualization
    st.subheader("📈 Model Metrics Comparison")
    fig_metrics = plot_metrics_comparison(results_dict)
    st.pyplot(fig_metrics)

    # 3. ROC Curves
    st.subheader("🎯 ROC Curves Analysis")
    fig_roc = plot_roc_curves(results_dict, class_names)
    st.pyplot(fig_roc)

    # 4. Individual Model Analysis
    st.subheader("🔍 Individual Model Analysis")

    for model_name, results in results_dict.items():
        full_name = MODEL_NAMES[model_name]

        with st.expander(f"{full_name} - Detailed Analysis", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Training Accuracy", f"{results['train_acc']:.4f}")
                st.metric("Training Loss", f"{results['train_loss']:.4f}")

            with col2:
                st.metric("Validation Accuracy", f"{results['val_acc']:.4f}")
                st.metric("Validation Loss", f"{results['val_loss']:.4f}")

            with col3:
                st.metric("Test Accuracy", f"{results['test_acc']:.4f}")
                st.metric("Test Loss", f"{results['test_loss']:.4f}")

            with col4:
                st.metric("ROC AUC", f"{results['metrics']['roc_auc']:.4f}")
                st.metric("F1-Score", f"{results['metrics']['f1_score']:.4f}")

            # Training History
            st.write("**Training Dynamics:**")
            fig_history = plot_training_history(results['history'], full_name)
            st.pyplot(fig_history)

            # Confusion Matrix
            st.write("**Classification Performance:**")
            col1, col2 = st.columns(2)

            with col1:
                fig_cm = plot_confusion_matrix(results['confusion_matrix'], class_names, full_name)
                st.pyplot(fig_cm)

            with col2:
                st.write("**Classification Report:**")
                report_df = pd.DataFrame(results['classification_report']).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    # 5. Best Model Recommendation
    best_model = max(results_dict.keys(), key=lambda x: results_dict[x]['test_acc'])
    best_accuracy = results_dict[best_model]['test_acc']
    best_model_name = MODEL_NAMES[best_model]

    st.success(f"""
    🏆 **Best Performing Model**: **{best_model_name}** 
    🎯 **Test Accuracy**: **{best_accuracy:.4f}**
    📊 **ROC AUC**: **{results_dict[best_model]['metrics']['roc_auc']:.4f}**
    """)


def preprocess_image(image):
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
    Compare multiple neural network models with comprehensive performance analysis.
    """)

    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'class_names' not in st.session_state:
        st.session_state.class_names = ['normal', 'stone']

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Train Models", "Predict"])

    with tab1:
        st.header("Welcome")
        st.markdown("""
        This app demonstrates comprehensive kidney stone detection using deep learning.

        **Models:**
        - Deep Neural Network (DNN)
        - Multi-Layer Perceptron (MLP) 
        - Autoencoder-based DNN

        **Comprehensive Analysis Includes:**
        - Training/Validation dynamics
        - Multiple performance metrics
        - ROC curves and AUC scores
        - Confusion matrices
        - Classification reports
        - Model comparison visualizations

        **How to use:**
        1. Add kidney images to dataset/stone/ and normal images to dataset/normal/
        2. Go to Train Models tab and click 'Start Training'
        3. View comprehensive analysis and use Predict tab for classification
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

                    st.session_state.class_names = class_names

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

                        # Comprehensive evaluation
                        results = evaluate_model_comprehensive(model, train_flat, val_flat, test_flat, model_name,
                                                               class_names)

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
                    st.success("All models trained successfully! Displaying comprehensive analysis...")
                    st.rerun()
                else:
                    st.error("Training failed for all models!")

        else:
            # Display comprehensive results
            display_comprehensive_results(st.session_state.results, st.session_state.class_names)

            if st.button("🔄 Retrain Models"):
                st.session_state.trained = False
                st.session_state.models = {}
                st.session_state.results = {}
                st.rerun()

    with tab3:
        st.header("Image Prediction")

        if not st.session_state.trained:
            st.info("Please train models first in the 'Train Models' tab!")
        else:
            uploaded = st.file_uploader("Upload kidney ultrasound image", type=['jpg', 'jpeg', 'png'])

            if uploaded:
                img = Image.open(uploaded)
                col1, col2 = st.columns(2)

                with col1:
                    st.image(img, caption="Uploaded Image", use_column_width=True)

                if st.button("Classify Image", type="primary"):
                    with st.spinner("Analyzing image..."):
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
                            st.subheader("Prediction Results")

                            for model_name, pred in predictions.items():
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h3 style="color: #dc3545; margin: 0; border-bottom: 2px solid #dc3545; padding-bottom: 0.5rem;">{model_name}</h3>
                                    <p style="margin: 0.5rem 0;"><strong style="color: #dc3545;">Prediction:</strong> <span class="metric-white">{pred['prediction']}</span></p>
                                    <p style="margin: 0.5rem 0;"><strong style="color: #dc3545;">Confidence:</strong> <span class="metric-white">{pred['confidence']:.4f}</span></p>
                                    <p style="margin: 0.5rem 0;"><strong style="color: #dc3545;">Stone Probability:</strong> <span class="metric-white">{pred['stone_prob']:.4f}</span></p>
                                    <p style="margin: 0.5rem 0;"><strong style="color: #dc3545;">Normal Probability:</strong> <span class="metric-white">{pred['normal_prob']:.4f}</span></p>
                                </div>
                                """, unsafe_allow_html=True)

                        # Consensus analysis
                        stone_votes = sum(1 for p in predictions.values() if p['prediction'] == 'Stone')
                        total = len(predictions)
                        consensus = "Stone" if stone_votes > total / 2 else "Normal"
                        consensus_color = "#dc3545" if consensus == "Stone" else "#28a745"

                        st.success(f"""
                        🎯 **Final Consensus: {consensus}** 
                        📊 **Agreement:** {stone_votes}/{total} models
                        """)

                        # Confidence summary
                        st.subheader("Model Confidence Summary")
                        conf_data = {
                            'Model': list(predictions.keys()),
                            'Confidence': [f"{pred['confidence']:.4f}" for pred in predictions.values()],
                            'Prediction': [pred['prediction'] for pred in predictions.values()],
                            'Stone Prob': [f"{pred['stone_prob']:.4f}" for pred in predictions.values()],
                            'Normal Prob': [f"{pred['normal_prob']:.4f}" for pred in predictions.values()]
                        }
                        conf_df = pd.DataFrame(conf_data)

                        # Style the dataframe
                        def style_predictions(val):
                            color = '#dc3545' if val == 'Stone' else '#28a745'
                            return f'color: {color}; font-weight: bold;'

                        styled_df = conf_df.style.applymap(style_predictions, subset=['Prediction'])
                        st.dataframe(styled_df, use_container_width=True)


if __name__ == "__main__":
    main()
