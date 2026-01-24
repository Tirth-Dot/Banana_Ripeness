"""
Demo Script - Shows you what to do next!
Run this to see the instructions and check your setup.
"""

import os
import sys

def check_requirements():
    """Check if required packages are installed"""
    print("="*60)
    print("CHECKING REQUIREMENTS")
    print("="*60)
    
    required = [
        'xgboost', 'streamlit', 'pandas', 'numpy', 
        'PIL', 'cv2', 'sklearn', 'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All requirements installed!")
        return True


def check_dataset():
    """Check if dataset is downloaded"""
    print("\n" + "="*60)
    print("CHECKING DATASET")
    print("="*60)
    
    data_folder = "data/images"
    labels_csv = "data/labels.csv"
    
    if os.path.exists(data_folder) and os.path.exists(labels_csv):
        num_images = len([f for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"[OK] Dataset found!")
        print(f"   Images folder: {data_folder}")
        print(f"   Labels CSV: {labels_csv}")
        print(f"   Number of images: {num_images}")
        return True
    else:
        print("[MISSING] Dataset not found!")
        print("\n[DOWNLOAD] DOWNLOAD THE DATASET:")
        print("   Option 1: python download_dataset.py")
        print("   Option 2: Manual download from Kaggle")
        print("   URL: https://www.kaggle.com/datasets/anishkumar00/days-death-to-a-banana")
        return False


def check_model():
    """Check if model is trained"""
    print("\n" + "="*60)
    print("CHECKING MODEL")
    print("="*60)
    
    model_file = "banana_xgboost_model.json"
    features_file = "feature_columns.pkl"
    
    if os.path.exists(model_file) and os.path.exists(features_file):
        print("[OK] Model found!")
        print(f"   Model: {model_file}")
        print(f"   Features: {features_file}")
        return True
    else:
        print("[MISSING] Model not trained yet!")
        print("\n[TRAIN] TRAIN THE MODEL:")
        print("   python train_model.py")
        return False


def main():
    """Main demo function"""
    print("\n")
    print("=" * 60)
    print("BANANA RIPENESS PREDICTOR - SETUP CHECK")
    print("=" * 60)
    print("\n")
    
    # Check requirements
    requirements_ok = check_requirements()
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Check model
    model_ok = check_model()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY & NEXT STEPS")
    print("="*60)
    
    if not requirements_ok:
        print("\n[1] Step 1: Install dependencies")
        print("   pip install -r requirements.txt")
    
    if not dataset_ok:
        print("\n[2] Step 2: Download dataset")
        print("   python download_dataset.py")
        print("   OR manually from Kaggle")
    
    if not model_ok and dataset_ok:
        print("\n[3] Step 3: Train the model")
        print("   python train_model.py")
    
    if model_ok:
        print("\n[4] Step 4: Run the app!")
        print("   streamlit run app.py")
        print("\n[OK] Everything is ready! You can run the Streamlit app now.")
    else:
        print("\n[!] Complete the steps above first, then run:")
        print("   streamlit run app.py")
    
    print("\n" + "="*60)
    print("[HELP] For detailed help, see:")
    print("   - README.md (full documentation)")
    print("   - QUICKSTART.md (quick guide)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
