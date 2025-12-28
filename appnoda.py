"""
Music Genre Classification using Spectrogram Images
Streamlit App with Pre-trained CNN + LSTM + Temporal Attention Models
(Dataset Explorer and Model Evaluation removed)

Run with: streamlit run app.py
"""

import os
import warnings
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =============================================================================
# TENSORFLOW CONFIGURATION - Must be before importing TensorFlow
# =============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, BatchNormalization, Dropout, GaussianNoise,
    Permute, Reshape, Dense, LSTM, Bidirectional
)
# Note: sklearn imports (train_test_split, confusion_matrix, classification_report) 
# are removed as they are no longer used for the evaluation section.

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODEL PATHS - Automatically resolve paths relative to this script
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_A_PATH = os.path.join(SCRIPT_DIR, "model_a_custom_cnn_lstm_attention.h5")
MODEL_B_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "model_b_efficientnet_lstm_attention.weights.h5")

# Dataset path (optional - originally for dataset explorer)
# DATASET_ROOT = "Data"
# IMAGES_DIR = os.path.join(DATASET_ROOT, "images_original")

# =============================================================================
# CONSTANTS
# =============================================================================
IMG_HEIGHT = 128
IMG_WIDTH = 256
BATCH_SIZE = 16

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
NUM_CLASSES = len(GENRES)

# Genre colors for visualization
GENRE_COLORS = {
    'blues': '#1f77b4',
    'classical': '#ff7f0e',
    'country': '#2ca02c',
    'disco': '#d62728',
    'hiphop': '#9467bd',
    'jazz': '#8c564b',
    'metal': '#e377c2',
    'pop': '#7f7f7f',
    'reggae': '#bcbd22',
    'rock': '#17becf'
}

# =============================================================================
# GPU MEMORY MANAGEMENT
# =============================================================================
@st.cache_resource
def configure_gpu():
    """Configure GPU memory growth to prevent crashes."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"✅ GPU enabled ({len(gpus)} device(s))"
        except RuntimeError as e:
            return f"⚠️ GPU config failed: {e}"
    return "ℹ️ Using CPU"

# =============================================================================
# TEMPORAL ATTENTION LAYER - IMPROVED with Multi-Head Support
# =============================================================================
# Check if already registered to avoid error on Streamlit re-runs
try:
    @keras.utils.register_keras_serializable(package="Custom", name="TemporalAttention")
    class TemporalAttention(layers.Layer):
        """
        IMPROVED Temporal Attention Layer with optional multi-head attention.
        
        Given LSTM outputs of shape (batch, time_steps, features),
        this layer:
        1. Computes attention scores for each time step (with multiple heads)
        2. Applies softmax to get attention weights
        3. Returns weighted sum of the sequence (context vector)
        """
        
        def __init__(self, num_heads=4, **kwargs):
            super(TemporalAttention, self).__init__(**kwargs)
            self.num_heads = num_heads
        
        def build(self, input_shape):
            self.features_dim = input_shape[-1]
            
            # Multi-head attention scoring
            self.attention_heads = []
            for i in range(self.num_heads):
                head = Dense(
                    units=self.features_dim // self.num_heads,
                    activation='tanh',
                    use_bias=True,
                    name=f'attention_head_{i}'
                )
                self.attention_heads.append(head)
            
            # Score projection for each head
            self.score_projections = []
            for i in range(self.num_heads):
                proj = Dense(units=1, use_bias=False, name=f'score_proj_{i}')
                self.score_projections.append(proj)
            
            # Final combination layer
            self.combine_layer = Dense(
                units=self.features_dim,
                activation='relu',
                name='attention_combine'
            )
            
            super(TemporalAttention, self).build(input_shape)
        
        def call(self, inputs, mask=None):
            # inputs shape: (batch, time_steps, features)
            
            context_vectors = []
            all_attention_weights = []
            
            for i in range(self.num_heads):
                # Compute attention for each head
                head_features = self.attention_heads[i](inputs)
                attention_scores = self.score_projections[i](head_features)
                attention_scores = tf.squeeze(attention_scores, axis=-1)
                attention_weights = tf.nn.softmax(attention_scores, axis=-1)
                
                attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
                context = tf.reduce_sum(inputs * attention_weights_expanded, axis=1)
                
                context_vectors.append(context)
                all_attention_weights.append(attention_weights)
            
            # Concatenate all heads and combine
            combined_context = tf.concat(context_vectors, axis=-1)
            final_context = self.combine_layer(combined_context)
            
            # Average attention weights for visualization
            avg_attention_weights = tf.reduce_mean(tf.stack(all_attention_weights, axis=0), axis=0)
            
            return final_context, avg_attention_weights
        
        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])
        
        def get_config(self):
            config = super(TemporalAttention, self).get_config()
            config.update({'num_heads': self.num_heads})
            return config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
except ValueError:
    # Already registered, get from registry
    pass

# =============================================================================
# MODEL BUILDER - Model B Only
# =============================================================================
# Model A: Loaded directly from .h5 file (no rebuild needed)
# Model B: Saved as weights only (.weights.h5) -> MUST rebuild architecture

def build_model_b(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES, fine_tune_from=100):
    """
    Build OPTIMIZED Model B: EfficientNet-B0 + LSTM + Multi-Head Temporal Attention.
    
    REQUIRED: Model B was saved as weights-only, so this architecture
    MUST be rebuilt before loading the saved weights.
    
    NOTE: This architecture MUST match exactly what was used during training.
    
    Architecture (matching notebook):
    - GaussianNoise input regularization
    - EfficientNet-B0 backbone (fine-tuned from layer 100)
    - Light backbone dropout (0.15)
    - Feature reduction with L2=0.002
    - BiLSTM layers: 160, 80 units with BatchNorm
    - Multi-head temporal attention (4 heads)
    - Classification head: 256, 128 units
    """
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GaussianNoise
    
    inputs = Input(shape=input_shape, name='input_spectrogram')
    
    # Input regularization - Light noise (matching notebook)
    x = GaussianNoise(0.01, name='input_noise')(inputs)
    
    # EfficientNet preprocessing
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    
    # EfficientNet backbone
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    
    # Fine-tuning: unfreeze from layer 100 (matching notebook)
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    
    backbone_output = base_model.output
    
    # Light dropout after backbone (matching notebook)
    backbone_output = Dropout(0.15, name='backbone_dropout')(backbone_output)
    
    # Get output shape dynamically
    height = base_model.output_shape[1]
    width = base_model.output_shape[2]
    filters = base_model.output_shape[3]
    
    # Reshape for LSTM
    x = Permute((2, 1, 3), name='permute_to_time_first')(backbone_output)
    x = Reshape((width, height * filters), name='reshape_to_sequence')(x)
    
    # Feature reduction - matching notebook (L2=0.002)
    l2_reg = keras.regularizers.l2(0.002)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2_reg, name='feature_reduction1')(x)
    x = BatchNormalization(name='bn_reduction1')(x)
    x = Dropout(0.25, name='dropout_reduction1')(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2_reg, name='feature_reduction2')(x)
    x = BatchNormalization(name='bn_reduction2')(x)
    x = Dropout(0.25, name='dropout_reduction2')(x)
    
    # LSTM layers - matching notebook (160, 80 units with BatchNorm)
    x = Bidirectional(LSTM(160, return_sequences=True, dropout=0.25, recurrent_dropout=0.1), name='bilstm_1')(x)
    x = BatchNormalization(name='bn_lstm1')(x)
    
    x = Bidirectional(LSTM(80, return_sequences=True, dropout=0.25, recurrent_dropout=0.1), name='bilstm_2')(x)
    x = BatchNormalization(name='bn_lstm2')(x)
    
    # Multi-Head Temporal Attention
    context_vector, attention_weights = TemporalAttention(num_heads=4, name='temporal_attention')(x)
    
    # Classification head - matching notebook (256, 128 units)
    x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002), name='fc1')(context_vector)
    x = BatchNormalization(name='bn_fc1')(x)
    x = Dropout(0.4, name='dropout_fc1')(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002), name='fc2')(x)
    x = BatchNormalization(name='bn_fc2')(x)
    x = Dropout(0.3, name='dropout_fc2')(x)
    
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='EfficientNet_LSTM_Attention_Optimized')
    return model

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================
# Model A: Full model was saved (.h5) -> Load directly without rebuilding
# Model B: Only weights were saved (.weights.h5) -> Must rebuild architecture then load weights

@st.cache_resource
def load_model_a():
    """
    Load Model A directly from saved .h5 file.
    Model A was saved as a full model, so we can load it without rebuilding.
    """
    if not os.path.exists(MODEL_A_PATH):
        return None, f"File not found: {MODEL_A_PATH}"
    
    try:
        # Load the full saved model with custom objects
        model = tf.keras.models.load_model(
            MODEL_A_PATH,
            custom_objects={'TemporalAttention': TemporalAttention},
            compile=False
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model, None
    except Exception as e:
        return None, f"Failed to load Model A: {str(e)}"


@st.cache_resource
def load_model_b():
    """
    Load Model B by rebuilding architecture and loading weights.
    Model B was saved as weights-only (.weights.h5), so we must rebuild the architecture first.
    """
    if not os.path.exists(MODEL_B_WEIGHTS_PATH):
        return None, f"File not found: {MODEL_B_WEIGHTS_PATH}"
    
    try:
        # Must rebuild the model architecture since only weights were saved
        # fine_tune_from=100 matches the notebook training configuration
        model = build_model_b(fine_tune_from=100)
        
        # Initialize model by running a dummy prediction (ensures all layers are built)
        dummy_input = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        _ = model(dummy_input, training=False)
        
        # Load the saved weights
        model.load_weights(MODEL_B_WEIGHTS_PATH)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002),  # Match notebook LR
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model, None
    except Exception as e:
        return None, f"Failed to load Model B weights: {str(e)}"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# compute_file_hash removed

def preprocess_image_custom(image_path):
    """Preprocess image for custom CNN model (normalized to [0,1])."""
    img = keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_efficientnet(image_path):
    """Preprocess image for EfficientNet model (keeps [0,255])."""
    img = keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras.utils.img_to_array(img)
    return np.expand_dims(img_array, axis=0)


def predict_genre(model, image_path, model_type='custom'):
    """Predict the genre of a single spectrogram image."""
    if model_type == 'custom':
        img_array = preprocess_image_custom(image_path)
    else:
        img_array = preprocess_image_efficientnet(image_path)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_genre = GENRES[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    return predicted_genre, predictions[0], confidence


# load_dataset_info removed
# safe_split_data removed

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_prediction_bar(predictions, predicted_genre):
    """Create a horizontal bar chart of prediction probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [GENRE_COLORS[g] if g == predicted_genre else '#cccccc' for g in GENRES]
    
    y_pos = np.arange(len(GENRES))
    bars = ax.barh(y_pos, predictions, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([g.capitalize() for g in GENRES])
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Genre Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    for i, (bar, prob) in enumerate(zip(bars, predictions)):
        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comparison_predictions(pred_a, pred_b, genre_a, genre_b):
    """Create side-by-side comparison of both models' predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    y_pos = np.arange(len(GENRES))
    
    # Model A
    colors_a = [GENRE_COLORS[g] if g == genre_a else '#cccccc' for g in GENRES]
    axes[0].barh(y_pos, pred_a, color=colors_a, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([g.capitalize() for g in GENRES])
    axes[0].set_xlabel('Probability')
    axes[0].set_title(f'Model A: {genre_a.capitalize()}', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Model B
    colors_b = [GENRE_COLORS[g] if g == genre_b else '#cccccc' for g in GENRES]
    axes[1].barh(y_pos, pred_b, color=colors_b, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([g.capitalize() for g in GENRES])
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f'Model B: {genre_b.capitalize()}', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


# plot_confusion_matrix_fig removed
# display_genre_samples removed

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Configure GPU
    gpu_status = configure_gpu()
    
    # Header
    st.title("🎵 Music Genre Classification")
    st.markdown("""
    **Deep Learning with CNN + LSTM + Temporal Attention**
    
    Classify music genres from Mel spectrogram images using pre-trained models.
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown(f"**TensorFlow:** {tf.__version__}")
    st.sidebar.markdown(f"**Status:** {gpu_status}")
    
    # =============================================================================
    # LOAD MODELS AT STARTUP
    # =============================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Model Status")
    
    # Load Model A
    model_a, error_a = load_model_a()
    
    if model_a:
        st.sidebar.success("✅ Model A loaded")
    else:
        st.sidebar.error(f"❌ Model A: {error_a}")
    
    # Load Model B
    model_b, error_b = load_model_b()
    
    if model_b:
        if error_b:
            st.sidebar.warning(f"⚠️ Model B: {error_b}")
        else:
            st.sidebar.success("✅ Model B loaded")
    else:
        st.sidebar.error(f"❌ Model B: {error_b}")
    
    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📍 Navigation",
        ["🏠 Home", "🔮 Predict Genre"] # Removed "📊 Dataset Explorer" and "📈 Model Evaluation"
    )
    
    # =============================================================================
    # HOME PAGE
    # =============================================================================
    if page == "🏠 Home":
        st.header("Welcome! 👋")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model A: Custom CNN")
            st.markdown("""
            - 4 Convolutional blocks with asymmetric pooling
            - Bidirectional LSTM layers
            - Temporal attention mechanism
            - BatchNorm & Dropout regularization
            """)
            if model_a:
                st.info(f"**Parameters:** {model_a.count_params():,}")
        
        with col2:
            st.subheader("🤖 Model B: EfficientNet")
            st.markdown("""
            - Pre-trained EfficientNet-B0 backbone
            - Transfer learning from ImageNet
            - Bidirectional LSTM layers
            - Temporal attention mechanism
            """)
            if model_b:
                st.info(f"**Parameters:** {model_b.count_params():,}")
        
        st.markdown("---")
        
        st.subheader("🎵 Supported Genres")
        cols = st.columns(5)
        for i, genre in enumerate(GENRES):
            with cols[i % 5]:
                color = GENRE_COLORS[genre]
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>● {genre.capitalize()}</span>", 
                            unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Go to "Predict Genre"** in the sidebar
        2. **Upload a spectrogram image** (PNG format)
        3. **Click "Predict"** to see genre predictions from both models
        4. **Compare results** between Model A and Model B
        """)
        
        # Model paths info
        with st.expander("📁 Model Paths"):
            st.code(f"Model A: {MODEL_A_PATH}")
            st.code(f"Model B: {MODEL_B_WEIGHTS_PATH}")
    
    # =============================================================================
    # PREDICT GENRE PAGE
    # =============================================================================
    elif page == "🔮 Predict Genre":
        st.header("🔮 Genre Prediction")
        
        # Check model status
        if not model_a and not model_b:
            st.error("❌ No models loaded! Please check the model paths.")
            st.stop()
        
        # Model status display
        col1, col2 = st.columns(2)
        with col1:
            if model_a:
                st.success("✅ Model A: Ready")
            else:
                st.warning("⚠️ Model A: Not available")
        with col2:
            if model_b:
                st.success("✅ Model B: Ready")
            else:
                st.warning("⚠️ Model B: Not available")
        
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Upload a Mel Spectrogram Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a spectrogram image (PNG recommended)"
        )
        
        if uploaded_file:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
            
            # Layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                img = keras.utils.load_img(temp_path)
                st.image(img, use_container_width=True)
                
                # Image info
                img_array = keras.utils.img_to_array(img)
                st.caption(f"Original size: {img_array.shape[1]}x{img_array.shape[0]}")
                st.caption(f"Will be resized to: {IMG_WIDTH}x{IMG_HEIGHT}")
            
            with col2:
                st.subheader("🎯 Prediction Results")
                
                # Model selection
                available_models = []
                if model_a:
                    available_models.append("Model A (Custom CNN)")
                if model_b:
                    available_models.append("Model B (EfficientNet)")
                if model_a and model_b:
                    available_models.append("Both Models")
                
                model_choice = st.radio(
                    "Select Model",
                    available_models,
                    horizontal=True
                )
                
                if st.button("🔮 Predict Genre", type="primary", use_container_width=True):
                    results = {}
                    
                    # Model A prediction
                    if model_a and model_choice in ["Model A (Custom CNN)", "Both Models"]:
                        with st.spinner("Model A analyzing..."):
                            genre_a, probs_a, conf_a = predict_genre(model_a, temp_path, 'custom')
                            results['a'] = (genre_a, probs_a, conf_a)
                    
                    # Model B prediction
                    if model_b and model_choice in ["Model B (EfficientNet)", "Both Models"]:
                        with st.spinner("Model B analyzing..."):
                            genre_b, probs_b, conf_b = predict_genre(model_b, temp_path, 'efficientnet')
                            results['b'] = (genre_b, probs_b, conf_b)
                    
                    # Display results
                    st.markdown("---")
                    
                    if 'a' in results and 'b' in results:
                        # Both models - side by side
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.markdown("### Model A")
                            color_a = GENRE_COLORS[results['a'][0]]
                            st.markdown(f"<h2 style='color:{color_a};'>{results['a'][0].upper()}</h2>", 
                                        unsafe_allow_html=True)
                            st.metric("Confidence", f"{results['a'][2]:.1%}")
                        
                        with res_col2:
                            st.markdown("### Model B")
                            color_b = GENRE_COLORS[results['b'][0]]
                            st.markdown(f"<h2 style='color:{color_b};'>{results['b'][0].upper()}</h2>", 
                                        unsafe_allow_html=True)
                            st.metric("Confidence", f"{results['b'][2]:.1%}")
                        
                        # Agreement check
                        if results['a'][0] == results['b'][0]:
                            st.success(f"✅ Both models agree: **{results['a'][0].upper()}**")
                        else:
                            st.warning(f"⚠️ Models disagree: A={results['a'][0].upper()}, B={results['b'][0].upper()}")
                        
                        # Comparison chart
                        fig = plot_comparison_predictions(
                            results['a'][1], results['b'][1],
                            results['a'][0], results['b'][0]
                        )
                        st.pyplot(fig)
                        plt.close()
                        
                    elif 'a' in results:
                        color = GENRE_COLORS[results['a'][0]]
                        st.markdown(f"<h1 style='color:{color}; text-align:center;'>{results['a'][0].upper()}</h1>", 
                                    unsafe_allow_html=True)
                        st.metric("Confidence", f"{results['a'][2]:.1%}")
                        fig = plot_prediction_bar(results['a'][1], results['a'][0])
                        st.pyplot(fig)
                        plt.close()
                        
                    elif 'b' in results:
                        color = GENRE_COLORS[results['b'][0]]
                        st.markdown(f"<h1 style='color:{color}; text-align:center;'>{results['b'][0].upper()}</h1>", 
                                    unsafe_allow_html=True)
                        st.metric("Confidence", f"{results['b'][2]:.1%}")
                        fig = plot_prediction_bar(results['b'][1], results['b'][0])
                        st.pyplot(fig)
                        plt.close()
                        
                    # Top 3 predictions
                    st.markdown("---")
                    st.subheader("📊 Top 3 Predictions")
                    
                    if 'a' in results:
                        top3_a = np.argsort(results['a'][1])[::-1][:3]
                        st.markdown("**Model A:**")
                        for i, idx in enumerate(top3_a):
                            st.write(f"{i+1}. {GENRES[idx].capitalize()}: {results['a'][1][idx]:.1%}")
                    
                    if 'b' in results:
                        top3_b = np.argsort(results['b'][1])[::-1][:3]
                        st.markdown("**Model B:**")
                        for i, idx in enumerate(top3_b):
                            st.write(f"{i+1}. {GENRES[idx].capitalize()}: {results['b'][1][idx]:.1%}")
    
# =============================================================================
# RUN APP
# =============================================================================
if __name__ == "__main__":
    main()