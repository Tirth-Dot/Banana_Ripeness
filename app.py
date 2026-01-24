"""
Banana Ripeness Prediction - Streamlit App
Upload a banana image to predict how many days until it goes bad!
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import xgboost as xgb
import joblib
import os
import io


class ImageFeatureExtractor:
    """Extract relevant features from banana images for XGBoost"""
    
    @staticmethod
    def extract_color_features(image):
        """Extract color-based features from the image
        
        Args:
            image: PIL Image or numpy array
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # If grayscale, convert to BGR
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Convert to different color spaces
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Extract mean and std for each channel
            features = {}
            
            # RGB features
            for i, channel in enumerate(['R', 'G', 'B']):
                features[f'mean_{channel}'] = np.mean(img_rgb[:, :, i])
                features[f'std_{channel}'] = np.std(img_rgb[:, :, i])
            
            # HSV features (very important for ripeness - Hue, Saturation, Value)
            for i, channel in enumerate(['H', 'S', 'V']):
                features[f'mean_{channel}'] = np.mean(img_hsv[:, :, i])
                features[f'std_{channel}'] = np.std(img_hsv[:, :, i])
            
            # LAB features (L=lightness, a=green-red, b=blue-yellow)
            for i, channel in enumerate(['L', 'A', 'B_lab']):
                features[f'mean_{channel}'] = np.mean(img_lab[:, :, i])
                features[f'std_{channel}'] = np.std(img_lab[:, :, i])
            
            # Calculate ratios (important for banana ripeness)
            # Yellow bananas have higher R and G, lower B
            features['yellow_ratio'] = (features['mean_R'] + features['mean_G']) / (features['mean_B'] + 1)
            
            # Brown spots indicator (low saturation, mid value in HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, brown_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            features['brown_spot_ratio'] = np.sum(brown_mask > 0) / brown_mask.size
            
            # Texture features using edge detection
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            return features
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None


def load_model():
    """Load the trained XGBoost model"""
    try:
        model = xgb.XGBRegressor()
        model.load_model('banana_xgboost_model.json')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, feature_columns
    except FileNotFoundError:
        return None, None


def predict_banana_freshness(image, model, feature_columns):
    """Predict days until banana goes bad"""
    # Extract features
    features = ImageFeatureExtractor.extract_color_features(image)
    
    if features is None:
        return None, None
    
    # Convert to DataFrame with correct column order
    features_df = pd.DataFrame([features])[feature_columns]
    
    # Make prediction
    prediction = model.predict(features_df)[0]
    
    return prediction, features


def get_ripeness_category(days):
    """Get ripeness category and color based on days"""
    if days >= 5:
        return "Fresh & Green", "🟢", "#4CAF50"
    elif days >= 3:
        return "Ripe & Perfect", "🟡", "#FFC107"
    elif days >= 1:
        return "Very Ripe", "🟠", "#FF9800"
    else:
        return "Overripe", "🔴", "#F44336"


def main():
    # Page configuration
    st.set_page_config(
        page_title="Banana Ripeness Predictor",
        page_icon="🍌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        h1 {
            color: #FFD700;
            text-align: center;
            font-size: 3.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: white;
            font-size: 1.3rem;
            margin-top: 0;
            margin-bottom: 2rem;
        }
        .prediction-box {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .feature-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1>🍌 Banana Ripeness Predictor</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Freshness Detection using XGBoost</p>', 
                unsafe_allow_html=True)
    
    # Load model
    model, feature_columns = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running:")
        st.code("python train_model.py", language="bash")
        st.info("""
        **Steps to get started:**
        1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/anishkumar00/days-death-to-a-banana)
        2. Extract it to a `data` folder
        3. Run the training script: `python train_model.py`
        4. Return here and refresh the page!
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://em-content.zobj.net/source/apple/391/banana_1f34c.png", width=100)
        st.title("About")
        st.info("""
        This app uses **XGBoost** machine learning to predict how many days 
        until a banana goes bad based on visual features.
        
        **Features analyzed:**
        - Color distribution (RGB, HSV, LAB)
        - Yellow/brown ratios
        - Texture patterns
        - Brown spot detection
        """)
        
        st.title("How it works")
        st.markdown("""
        1. 📸 Upload a banana image
        2. 🔍 AI extracts visual features
        3. 🤖 XGBoost predicts freshness
        4. 📊 Get detailed analysis
        """)
        
        st.title("Model Info")
        st.metric("Total Features", len(feature_columns) if feature_columns else 0)
        st.success("✅ Model Loaded")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("📤 Upload Banana Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear photo of a banana"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔮 Predict Freshness", type="primary", use_container_width=True):
                with st.spinner("Analyzing banana..."):
                    prediction, features = predict_banana_freshness(image, model, feature_columns)
                
                if prediction is not None:
                    # Store in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['features'] = features
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            features = st.session_state['features']
            
            st.subheader("🎯 Prediction Results")
            
            # Get category
            category, emoji, color = get_ripeness_category(prediction)
            
            # Main prediction display
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: {color}; 
                     border-radius: 15px; margin: 20px 0;'>
                    <h2 style='color: white; margin: 0;'>{emoji} {category}</h2>
                    <h1 style='color: white; font-size: 4rem; margin: 10px 0;'>
                        {prediction:.1f} days
                    </h1>
                    <p style='color: white; font-size: 1.2rem; margin: 0;'>
                        until this banana goes bad
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if prediction >= 5:
                st.success("🟢 **Perfect for storage!** This banana is still quite fresh. Store at room temperature.")
            elif prediction >= 3:
                st.warning("🟡 **Ready to eat!** This banana is at peak ripeness. Enjoy it now or use in smoothies.")
            elif prediction >= 1:
                st.warning("🟠 **Eat soon!** This banana is very ripe. Great for banana bread or smoothies.")
            else:
                st.error("🔴 **Use immediately!** This banana is overripe. Best for baking or composting.")
            
            # Feature insights
            with st.expander("📊 View Detailed Analysis"):
                st.markdown("### Key Visual Features")
                
                # Create metrics in columns
                fcol1, fcol2, fcol3 = st.columns(3)
                
                with fcol1:
                    st.metric("Yellow Ratio", f"{features['yellow_ratio']:.2f}")
                    st.metric("Hue (Color)", f"{features['mean_H']:.1f}")
                
                with fcol2:
                    st.metric("Brown Spots", f"{features['brown_spot_ratio']:.2%}")
                    st.metric("Saturation", f"{features['mean_S']:.1f}")
                
                with fcol3:
                    st.metric("Edge Density", f"{features['edge_density']:.2%}")
                    st.metric("Brightness", f"{features['mean_V']:.1f}")
                
                # RGB values
                st.markdown("### RGB Color Analysis")
                rgb_col1, rgb_col2, rgb_col3 = st.columns(3)
                
                with rgb_col1:
                    st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);'>
                            <h3 style='color: white; margin: 0;'>Red</h3>
                            <h2 style='color: white;'>{features['mean_R']:.0f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with rgb_col2:
                    st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);'>
                            <h3 style='color: white; margin: 0;'>Green</h3>
                            <h2 style='color: white;'>{features['mean_G']:.0f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with rgb_col3:
                    st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                            <h3 style='color: white; margin: 0;'>Blue</h3>
                            <h2 style='color: white;'>{features['mean_B']:.0f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.info("👈 Upload a banana image to get started!")
            st.markdown("""
                ### What this app does:
                
                - Analyzes color patterns (RGB, HSV, LAB)
                - Detects brown spots and texture
                - Predicts remaining freshness days
                - Provides storage recommendations
                
                ### Tips for best results:
                
                - Use clear, well-lit photos
                - Ensure banana is the main subject
                - Avoid heavily filtered images
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: white; padding: 20px;'>
            <p style='font-size: 0.9rem;'>
                🤖 Powered by <b>XGBoost</b> | 
                Built with <b>Streamlit</b> | 
                Data from <a href='https://www.kaggle.com/datasets/anishkumar00/days-death-to-a-banana' 
                style='color: #FFD700;'>Kaggle</a>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
