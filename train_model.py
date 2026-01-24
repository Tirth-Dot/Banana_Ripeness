"""
Banana Ripeness Prediction - Training Script
This script extracts features from banana images and trains an XGBoost model
to predict the number of days until the banana goes bad.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")


class ImageFeatureExtractor:
    """Extract relevant features from banana images for XGBoost"""
    
    @staticmethod
    def extract_color_features(image_path):
        """Extract color-based features from the image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
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
            print(f"Error processing {image_path}: {e}")
            return None


def load_dataset(data_folder, labels_csv):
    """Load images and labels from the dataset"""
    print("Loading dataset...")
    
    # Read labels
    if not os.path.exists(labels_csv):
        print(f"Error: Labels file not found at {labels_csv}")
        print("Please ensure you have downloaded the dataset from Kaggle and extracted it.")
        return None, None
    
    labels_df = pd.read_csv(labels_csv)
    print(f"Loaded {len(labels_df)} image labels")
    
    # Detect column names once
    image_col = None
    for col in ['image_filename', 'filename', 'image_name', 'image', 'Image', 'Filename']:
        if col in labels_df.columns:
            image_col = col
            break
            
    label_col = None
    for col in ['days_left', 'days', 'label', 'target', 'days_to_death', 'Days']:
        if col in labels_df.columns:
            label_col = col
            break
            
    if image_col is None or label_col is None:
        print(f"Available columns: {labels_df.columns.tolist()}")
        print(f"Found image column: {image_col}, label column: {label_col}")
        
        # Try fallbacks if exact match failed but we can guess
        if image_col is None and 'image_filename' not in labels_df.columns:
             # Just take the first column if it looks like string
             if len(labels_df.columns) > 0:
                 image_col = labels_df.columns[0]
                 print(f"Warning: Guessing image column is {image_col}")
        
        if label_col is None and 'days_left' not in labels_df.columns:
             # Just take the second column if it looks numeric
             if len(labels_df.columns) > 1:
                 label_col = labels_df.columns[1]
                 print(f"Warning: Guessing label column is {label_col}")

        if image_col is None or label_col is None:
            print("ERROR: Could not determine data columns.")
            return None, None

    print(f"Using columns: Img='{image_col}', Target='{label_col}'")

    # Extract features from all images
    features_list = []
    labels_list = []
    
    for idx, row in labels_df.iterrows():
        image_name = row[image_col]
        days_to_death = row[label_col]
        
        # Construct full image path
        image_path = os.path.join(data_folder, image_name)
        
        # Fix potential path issues (e.g. if CSV has just filename but images are in subfolders? 
        # No, structure is flat in Data/images according to ls output)
        
        if not os.path.exists(image_path):
            # Try checking if there's a mismatch in extension or path
            # But based on `ls` output, filenames look correct.
            # print(f"Warning: Image not found: {image_path}")
            continue
        
        # Extract features
        features = ImageFeatureExtractor.extract_color_features(image_path)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(days_to_death)
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(labels_df)} images...")
    
    # Convert to DataFrame
    X = pd.DataFrame(features_list)
    y = np.array(labels_list)
    
    print(f"\nSuccessfully extracted features from {len(X)} images")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    
    return X, y


def train_xgboost_model(X, y):
    """Train XGBoost model for regression"""
    print("\n" + "="*60)
    print("Training XGBoost Model")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse'
    )
    
    # Train model
    print("\nTraining...")
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    print("\n" + "="*60)
    print("Model Performance")
    print("="*60)
    
    print("\nTraining Set:")
    print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    
    print("\nTest Set:")
    print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    # Feature importance
    print("\n" + "="*60)
    print("Top 10 Most Important Features")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance.head(15)['feature'], 
             feature_importance.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("\nFeature importance plot saved as 'feature_importance.png'")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Days to Death')
    plt.ylabel('Predicted Days to Death')
    plt.title('Predictions vs Actual Values')
    plt.tight_layout()
    plt.savefig('predictions_plot.png', dpi=150, bbox_inches='tight')
    print("Predictions plot saved as 'predictions_plot.png'")
    
    return model, feature_importance


def save_model(model, feature_columns):
    """Save the trained model and metadata"""
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    # Save model
    model_path = 'banana_xgboost_model.json'
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature columns
    joblib.dump(feature_columns, 'feature_columns.pkl')
    print(f"Feature columns saved to: feature_columns.pkl")
    
    print("\nModel files created successfully!")


def main():
    """Main training pipeline"""
    print("="*60)
    print("BANANA RIPENESS PREDICTION - XGBoost Training")
    print("="*60)
    
    # Dataset paths -MODIFY THESE ACCORDING TO YOUR DATASET LOCATION
    data_folder = "Data/images"  # Folder containing banana images
    labels_csv = "Data/labels.csv"  # CSV file with image names and days to death
    
    print(f"\nDataset Configuration:")
    print(f"  Images folder: {data_folder}")
    print(f"  Labels CSV: {labels_csv}")
    print("\nIMPORTANT: Please ensure you have:")
    print("  1. Downloaded the dataset from Kaggle")
    print("  2. Extracted it to the 'data' folder")
    print("  3. Updated the paths above if needed")
    print("="*60 + "\n")
    
    # Check if data exists
    if not os.path.exists(data_folder):
        print(f"\n[ERROR] Data folder not found: {data_folder}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/anishkumar00/days-death-to-a-banana")
        print("\nThen extract it and update the paths in this script.")
        return
    
    # Load and extract features
    X, y = load_dataset(data_folder, labels_csv)
    
    if X is None or y is None:
        print("\n[ERROR] Failed to load dataset. Please check the paths and CSV structure.")
        return
    
    # Train model
    model, feature_importance = train_xgboost_model(X, y)
    
    # Save model
    save_model(model, X.columns.tolist())
    
    print("\n" + "="*60)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now run the Streamlit app with:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
