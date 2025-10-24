import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report

# Set page configuration
st.set_page_config(
    page_title="Kidney Stone Classifier",
    page_icon="",
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
    /* Tab styling - clean minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid #e6e6e6;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        padding: 10px 16px;
        border: none;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class KidneyStoneModel:
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.input_dim = img_size[0] * img_size[1] * 3
        self.class_names = ['Normal', 'Stone']
        self.models = {}
        self.evaluation_results = {}

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
                st.error(f"Dataset directory '{data_dir}' not found!")
                st.info("""
                Please ensure your dataset is in the correct structure:
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

            test_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=456,
                image_size=self.img_size,
                batch_size=32
            )

            # Get class names
            self.class_names = train_ds.class_names
            st.success(f"Found classes: {self.class_names}")

            # Cache/prefetch for performance
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

            # Normalize and flatten
            normalization_layer = layers.Rescaling(1. / 255)

            def preprocess_ds(ds):
                return ds.map(lambda x, y: (tf.reshape(normalization_layer(x), [tf.shape(x)[0], -1]), y))

            train_ds_flat = preprocess_ds(train_ds)
            val_ds_flat = preprocess_ds(val_ds)
            test_ds_flat = preprocess_ds(test_ds)

            return train_ds_flat, val_ds_flat, test_ds_flat, self.class_names

        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None, None, None, None

    def train_models(self, train_ds, val_ds, test_ds, epochs=10):
        """Train all models on real data"""
        models_info = {
            'DNN': self.create_dnn_model(),
            'MLP': self.create_mlp_model(),
            'AE-DNN': self.create_autoencoder_dnn_model()
        }

        results = {}

        for model_name, model in models_info.items():
            st.write(f"Training {model_name} model...")

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

            # Evaluate model on validation and test sets
            val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

            # Get predictions for comprehensive evaluation
            y_true_val, y_pred_val, y_probs_val = self.get_predictions(model, val_ds)
            y_true_test, y_pred_test, y_probs_test = self.get_predictions(model, test_ds)

            # Calculate comprehensive metrics
            val_metrics = self.calculate_comprehensive_metrics(y_true_val, y_pred_val, y_probs_val)
            test_metrics = self.calculate_comprehensive_metrics(y_true_test, y_pred_test, y_probs_test)

            # Store results
            results[model_name] = {
                'model': model,
                'history': history.history,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'y_true_val': y_true_val,
                'y_pred_val': y_pred_val,
                'y_probs_val': y_probs_val,
                'y_true_test': y_true_test,
                'y_pred_test': y_pred_test,
                'y_probs_test': y_probs_test
            }

            # Save model
            model_path = f"{model_name.lower()}_model.keras"
            model.save(model_path)
            self.models[model_name] = model

            # Update progress
            progress_bar.progress(100)
            status_text.success(
                f"{model_name} trained - Val Accuracy: {val_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}")

        self.evaluation_results = results
        return results

    def get_predictions(self, model, dataset):
        """Get predictions from model"""
        y_true = []
        y_pred = []
        y_probs = []

        for x, y in dataset:
            y_true.extend(y.numpy())
            predictions = model.predict(x, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_probs.extend(predictions[:, 1])  # Probability of class 1 (Stone)

        return np.array(y_true), np.array(y_pred), np.array(y_probs)

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_probs):
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'fpr': fpr,
            'tpr': tpr
        }

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


def main():
    # Initialize model handler
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = KidneyStoneModel()

    model_handler = st.session_state.model_handler

    # Header
    st.markdown('<h1 class="main-header">Kidney Stone Classification System</h1>', unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Home",
        "Train Models",
        "Predict",
        "Model Info"
    ])

    with tab1:
        show_home()

    with tab2:
        train_models_interface(model_handler)

    with tab3:
        predict_interface(model_handler)

    with tab4:
        show_model_info(model_handler)


def show_home():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Welcome to the Kidney Stone Classification System

        This AI-powered application helps classify kidney ultrasound images into:
        - **Normal** - No kidney stones detected
        - **Stone** - Kidney stones present

        ### Quick Start:
        1. **Train Models**: Train AI models on your data
        2. **Predict**: Upload new images for classification
        3. **View Results**: Get instant classification with confidence scores

    
        - Supported formats: JPG, JPEG, PNG
        - Recommended: 100+ images per class

        ### Medical Application:
        - Assist radiologists in kidney stone detection
        - Quick preliminary screening
        - Educational tool for medical students
        """)

    with col2:
        # Check if dataset exists
        if os.path.exists("dataset"):
            st.success("Dataset folder exists")
            # Count images
            normal_count = len(os.listdir("dataset/Normal")) if os.path.exists("dataset/Normal") else 0
            stone_count = len(os.listdir("dataset/Stone")) if os.path.exists("dataset/Stone") else 0
            st.metric("Normal Images", normal_count)
            st.metric("Stone Images", stone_count)
        else:
            st.warning("No dataset found")

        # Check if models are trained
        if hasattr(st.session_state.model_handler, 'models') and st.session_state.model_handler.models:
            st.success("Models are trained")
            st.metric("Trained Models", len(st.session_state.model_handler.models))
        else:
            st.warning("No models trained")


def train_models_interface(model_handler):
    st.header("Train Classification Models")

    # Check if dataset exists
    if not os.path.exists("dataset"):
        st.error("No dataset found! Please ensure your dataset is in the 'dataset' folder.")
        return

    st.markdown("""
    Train three different neural network models for kidney stone classification.
    The models will be trained on your dataset.
    """)

    # Training parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        epochs = st.slider("Number of Epochs", min_value=5, max_value=50, value=10)
    with col2:
        img_size = st.selectbox("Image Size", [64, 128], index=0, format_func=lambda x: f"{x}x{x}")
        model_handler.img_size = (img_size, img_size)
        model_handler.input_dim = img_size * img_size * 3
    with col3:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    # Model selection
    st.subheader("Model Selection")
    selected_models = st.multiselect(
        "Choose models to train:",
        ["DNN", "MLP", "AE-DNN"],
        default=["DNN", "MLP", "AE-DNN"],
        help="Select which models to train"
    )

    if st.button("Start Training", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return

        with st.spinner("Loading dataset and training models... This may take several minutes."):
            try:
                # Load and preprocess data
                train_ds, val_ds, test_ds, class_names = model_handler.load_and_preprocess_data()

                if train_ds is None:
                    return

                # Filter models based on selection
                models_to_train = {k: v for k, v in {
                    'DNN': model_handler.create_dnn_model(),
                    'MLP': model_handler.create_mlp_model(),
                    'AE-DNN': model_handler.create_autoencoder_dnn_model()
                }.items() if k in selected_models}

                # Train models
                results = {}
                for model_name, model in models_to_train.items():
                    st.write(f"Training {model_name} model...")
                    progress_bar = st.progress(0)

                    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
                    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
                    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

                    # Get predictions and calculate metrics
                    y_true_val, y_pred_val, y_probs_val = model_handler.get_predictions(model, val_ds)
                    y_true_test, y_pred_test, y_probs_test = model_handler.get_predictions(model, test_ds)

                    val_metrics = model_handler.calculate_comprehensive_metrics(y_true_val, y_pred_val, y_probs_val)
                    test_metrics = model_handler.calculate_comprehensive_metrics(y_true_test, y_pred_test, y_probs_test)

                    results[model_name] = {
                        'model': model,
                        'history': history.history,
                        'val_accuracy': val_accuracy,
                        'val_loss': val_loss,
                        'test_accuracy': test_accuracy,
                        'test_loss': test_loss,
                        'val_metrics': val_metrics,
                        'test_metrics': test_metrics,
                        'y_true_val': y_true_val,
                        'y_pred_val': y_pred_val,
                        'y_probs_val': y_probs_val,
                        'y_true_test': y_true_test,
                        'y_pred_test': y_pred_test,
                        'y_probs_test': y_probs_test
                    }

                    model_path = f"{model_name.lower()}_model.keras"
                    model.save(model_path)
                    model_handler.models[model_name] = model

                    progress_bar.progress(100)
                    st.success(
                        f"{model_name} trained - Val Accuracy: {val_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}")

                model_handler.evaluation_results = results
                st.success("Training completed successfully!")

                # Display results
                display_training_results(results)

                # Store in session state
                st.session_state.models_trained = True
                st.session_state.training_results = results

            except Exception as e:
                st.error(f"Training failed: {str(e)}")


def display_training_results(results):
    st.subheader("Training Results")

    # 1. Comprehensive Performance Table
    st.subheader("Comprehensive Model Performance Metrics")

    metrics_data = []
    for model_name, result in results.items():
        val_metrics = result['val_metrics']
        test_metrics = result['test_metrics']

        metrics_data.append({
            'Model': model_name,
            'Val Accuracy': f"{result['val_accuracy']:.2%}",
            'Test Accuracy': f"{result['test_accuracy']:.2%}",
            'Val Precision': f"{val_metrics['precision']:.2%}",
            'Test Precision': f"{test_metrics['precision']:.2%}",
            'Val Recall': f"{val_metrics['recall']:.2%}",
            'Test Recall': f"{test_metrics['recall']:.2%}",
            'Val F1-Score': f"{val_metrics['f1_score']:.2%}",
            'Test F1-Score': f"{test_metrics['f1_score']:.2%}",
            'Val ROC AUC': f"{val_metrics['roc_auc']:.3f}",
            'Test ROC AUC': f"{test_metrics['roc_auc']:.3f}"
        })

    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)

    # 2. ROC Curve Comparison
    st.subheader("ROC Curve Comparison")

    fig_roc = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (model_name, result) in enumerate(results.items()):
        val_metrics = result['val_metrics']
        fig_roc.add_trace(go.Scatter(
            x=val_metrics['fpr'],
            y=val_metrics['tpr'],
            name=f'{model_name} (AUC = {val_metrics["roc_auc"]:.3f})',
            line=dict(color=colors[i], width=3)
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color='black', dash='dash'),
        name='Random Classifier'
    ))

    fig_roc.update_layout(
        title='ROC Curves - Validation Set',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # 3. Training Dynamics
    st.subheader("Training Dynamics Analysis")

    for model_name, result in results.items():
        col1, col2 = st.columns(2)

        with col1:
            # Accuracy progression
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                y=result['history']['accuracy'],
                name='Train Accuracy',
                line=dict(color='blue', width=3)
            ))
            fig_acc.add_trace(go.Scatter(
                y=result['history']['val_accuracy'],
                name='Val Accuracy',
                line=dict(color='red', width=3, dash='dash')
            ))
            fig_acc.update_layout(
                title=f'{model_name} - Accuracy',
                xaxis_title='Epoch',
                yaxis_title='Accuracy',
                height=400
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            # Loss progression
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=result['history']['loss'],
                name='Train Loss',
                line=dict(color='blue', width=3)
            ))
            fig_loss.add_trace(go.Scatter(
                y=result['history']['val_loss'],
                name='Val Loss',
                line=dict(color='red', width=3, dash='dash')
            ))
            fig_loss.update_layout(
                title=f'{model_name} - Loss',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    # 4. Confusion Matrices
    st.subheader("Confusion Matrices - Test Set")

    cols = st.columns(len(results))
    for i, (model_name, result) in enumerate(results.items()):
        with cols[i]:
            cm = result['test_metrics']['confusion_matrix']
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Blues',
                title=f'{model_name}'
            )
            fig_cm.update_layout(
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=300
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # 5. Performance Comparison
    st.subheader("Performance Comparison")

    # Accuracy comparison
    models = list(results.keys())
    val_accs = [results[model]['val_accuracy'] for model in models]
    test_accs = [results[model]['test_accuracy'] for model in models]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='Validation Accuracy', x=models, y=val_accs, marker_color='lightblue'))
    fig_comp.add_trace(go.Bar(name='Test Accuracy', x=models, y=test_accs, marker_color='lightcoral'))
    fig_comp.update_layout(title='Accuracy Comparison', barmode='group', height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

    # 6. Classification Reports
    st.subheader("Detailed Classification Reports - Test Set")

    for model_name, result in results.items():
        with st.expander(f"Classification Report - {model_name}"):
            class_report = result['test_metrics']['classification_report']
            report_df = pd.DataFrame(class_report).transpose()
            st.dataframe(report_df, use_container_width=True)


def predict_interface(model_handler):
    st.header("Predict Kidney Stone")

    # Check if models are trained
    if not hasattr(model_handler, 'models') or not model_handler.models:
        st.warning("No trained models found! Please train models first.")
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
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Analyze Image with All Models", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with all models..."):
                    try:
                        # Get predictions from all models
                        all_predictions = {}
                        for model_name in model_handler.models.keys():
                            prediction, confidence, probabilities = model_handler.predict_image(
                                image, model_name
                            )
                            all_predictions[model_name] = {
                                'prediction': prediction,
                                'confidence': confidence,
                                'probabilities': probabilities
                            }

                        # Display results in the right column
                        with col2:
                            display_all_predictions_results(
                                all_predictions, image
                            )

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")


def display_all_predictions_results(all_predictions, image):
    st.subheader("Analysis Results from All Models")

    # Create a summary table of all predictions
    st.subheader("Model Predictions Summary")

    summary_data = []
    for model_name, result in all_predictions.items():
        summary_data.append({
            'Model': model_name,
            'Prediction': result['prediction'],
            'Confidence': f"{result['confidence']:.2%}",
            'Normal Probability': f"{result['probabilities']['Normal']:.2%}",
            'Stone Probability': f"{result['probabilities']['Stone']:.2%}"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    # Display individual model results
    st.subheader("Detailed Model Analysis")

    # Create columns for model results (2 models per row)
    model_names = list(all_predictions.keys())
    num_models = len(model_names)

    for i in range(0, num_models, 2):
        cols = st.columns(2)

        for j, col in enumerate(cols):
            if i + j < num_models:
                model_name = model_names[i + j]
                result = all_predictions[model_name]

                with col:
                    display_single_model_result(model_name, result)

    # Consensus analysis
    display_consensus_analysis(all_predictions)


def display_single_model_result(model_name, result):
    """Display prediction results for a single model"""
    prediction = result['prediction']
    confidence = result['confidence']
    probabilities = result['probabilities']

    # Prediction result with color
    if prediction == "Stone":
        box_class = "stone"
    else:
        box_class = "normal"

    # Prediction box for individual model
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h3>{model_name}</h3>
        <h4>Prediction: {prediction}</h4>
        <p>Confidence: {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability chart for this model
    fig = go.Figure(go.Bar(
        x=list(probabilities.keys()),
        y=list(probabilities.values()),
        marker_color=['green', 'red'],
        text=[f'{prob:.2%}' for prob in probabilities.values()],
        textposition='auto',
    ))

    fig.update_layout(
        title=f"{model_name} - Class Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=250,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def display_consensus_analysis(all_predictions):
    """Display consensus analysis across all models"""
    st.subheader("Consensus Analysis")

    # Calculate consensus
    predictions = [result['prediction'] for result in all_predictions.values()]
    normal_count = predictions.count('Normal')
    stone_count = predictions.count('Stone')
    total_models = len(all_predictions)

    col1, col2, col3 = st.columns(3)

    with col1:
        if normal_count > stone_count:
            consensus = "Normal"
            st.success(f"Consensus: {consensus}")
        elif stone_count > normal_count:
            consensus = "Stone"
            st.error(f"Consensus: {consensus}")
        else:
            consensus = "Tie"
            st.warning(f"Consensus: {consensus}")

        st.metric("Agreement", f"{max(normal_count, stone_count)}/{total_models}")

    with col2:
        st.metric("Normal Votes", normal_count)

    with col3:
        st.metric("Stone Votes", stone_count)

    # Agreement visualization
    fig_consensus = go.Figure(go.Bar(
        x=['Normal', 'Stone'],
        y=[normal_count, stone_count],
        marker_color=['green', 'red'],
        text=[f'{normal_count}', f'{stone_count}'],
        textposition='auto',
    ))

    fig_consensus.update_layout(
        title="Model Voting Consensus",
        xaxis_title="Prediction",
        yaxis_title="Number of Models",
        height=300
    )
    st.plotly_chart(fig_consensus, use_container_width=True)

    # Final medical advice based on consensus
    st.subheader("Medical Guidance")

    if consensus == "Stone":
        advice = """
        **Medical Advice:**
        - Multiple models detected kidney stones
        - Consult a urologist for further evaluation
        - Drink plenty of water
        - Consider follow-up imaging studies
        - Monitor for symptoms like pain or blood in urine
        """
        st.warning(advice)
    elif consensus == "Normal":
        advice = """
        **Medical Advice:**
        - Majority of models found no kidney stones
        - No kidney stones detected in this image
        - Maintain healthy hydration
        - Regular check-ups recommended
        - Continue healthy lifestyle habits
        """
        st.success(advice)
    else:
        advice = """
        **Medical Advice:**
        - Models are divided in their predictions
        - Consider getting a second opinion
        - Additional imaging may be needed
        - Consult with a healthcare professional
        """
        st.info(advice)

    # Disclaimer
    st.info("""
    **Disclaimer:** This is a demonstration tool. For actual medical diagnosis, 
    always consult with qualified healthcare professionals and rely on comprehensive clinical evaluation.
    """)


def show_model_info(model_handler):
    st.header("Model Information")

    col1, col2 = st.columns([2, 1])

    with col1:
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

    with col2:
        # Show trained models status
        st.subheader("Model Status")
        if hasattr(model_handler, 'models') and model_handler.models:
            for model_name in model_handler.models.keys():
                st.success(f"{model_name}")
            st.metric("Total Models", len(model_handler.models))
        else:
            st.warning("No models trained")
            st.metric("Total Models", 0)

        # System information
        st.subheader("System Info")
        st.metric("TensorFlow", tf.__version__)
        st.metric("Streamlit", st.__version__)

        # Dataset info
        if os.path.exists("dataset"):
            normal_count = len(os.listdir("dataset/Normal")) if os.path.exists("dataset/Normal") else 0
            stone_count = len(os.listdir("dataset/Stone")) if os.path.exists("dataset/Stone") else 0
            st.subheader("Dataset Info")
            st.metric("Normal Images", normal_count)
            st.metric("Stone Images", stone_count)


if __name__ == "__main__":
    main()
