"""
Dataset Download Helper
This script helps you download the Kaggle dataset using the Kaggle API.

Requirements:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads kaggle.json
   - Place it in: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json (Windows)
"""

import os
import sys
import zipfile
import shutil


def check_kaggle_setup():
    """Check if Kaggle API is properly set up"""
    try:
        import kaggle
        print("✅ Kaggle library is installed")
        return True
    except ImportError:
        print("❌ Kaggle library not found")
        print("\nPlease install it with:")
        print("  pip install kaggle")
        return False


def download_dataset():
    """Download the banana dataset from Kaggle"""
    import kaggle
    
    dataset_name = "anishkumar00/days-death-to-a-banana"
    download_path = "./kaggle_download"
    
    print(f"\n📥 Downloading dataset: {dataset_name}")
    print(f"📁 Download location: {download_path}")
    
    try:
        # Create download directory
        os.makedirs(download_path, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        return download_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Created a Kaggle account")
        print("2. Set up your API token (kaggle.json)")
        print("3. Accepted the dataset's terms on Kaggle website")
        return None


def organize_dataset(download_path):
    """Organize the downloaded dataset into the correct structure"""
    print("\n📂 Organizing dataset...")
    
    # Create data directory
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Look for downloaded files
    if not os.path.exists(download_path):
        print(f"❌ Download path not found: {download_path}")
        return False
    
    # List downloaded files
    files = os.listdir(download_path)
    print(f"Downloaded files: {files}")
    
    # Try to find images folder and CSV file
    images_found = False
    csv_found = False
    
    for item in os.listdir(download_path):
        item_path = os.path.join(download_path, item)
        
        # Check if it's the images folder
        if os.path.isdir(item_path) and 'image' in item.lower():
            dest_images = os.path.join(data_dir, "images")
            if os.path.exists(dest_images):
                shutil.rmtree(dest_images)
            shutil.copytree(item_path, dest_images)
            print(f"✅ Copied images to {dest_images}")
            images_found = True
        
        # Check if it's a CSV file
        elif item.endswith('.csv'):
            dest_csv = os.path.join(data_dir, "labels.csv")
            shutil.copy2(item_path, dest_csv)
            print(f"✅ Copied labels to {dest_csv}")
            csv_found = True
    
    if not images_found or not csv_found:
        print("\n⚠️ Dataset structure might be different than expected.")
        print("Please manually organize the files into:")
        print("  data/images/  - folder with banana images")
        print("  data/labels.csv  - CSV file with labels")
        return False
    
    print("\n✅ Dataset organized successfully!")
    return True


def main():
    """Main function"""
    print("="*60)
    print("BANANA DATASET DOWNLOADER")
    print("="*60)
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n" + "="*60)
        print("SETUP INSTRUCTIONS")
        print("="*60)
        print("\n1. Install Kaggle API:")
        print("   pip install kaggle")
        print("\n2. Get your API credentials:")
        print("   - Visit: https://www.kaggle.com/settings")
        print("   - Click 'Create New API Token'")
        print("   - Save kaggle.json to: C:\\Users\\<YourUsername>\\.kaggle\\")
        print("\n3. Run this script again")
        return
    
    # Download dataset
    download_path = download_dataset()
    
    if download_path:
        # Organize dataset
        organize_dataset(download_path)
        
        # Clean up
        print("\n🧹 Cleaning up temporary files...")
        try:
            shutil.rmtree(download_path)
            print("✅ Cleanup complete")
        except:
            print(f"⚠️ Could not remove {download_path}, you can delete it manually")
        
        print("\n" + "="*60)
        print("✅ DATASET READY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Verify the data folder structure:")
        print("   - data/images/  (should contain banana images)")
        print("   - data/labels.csv  (should contain labels)")
        print("\n2. Train the model:")
        print("   python train_model.py")
        print("\n3. Run the app:")
        print("   streamlit run app.py")
    else:
        print("\n" + "="*60)
        print("❌ DOWNLOAD FAILED")
        print("="*60)
        print("\nManual download instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/anishkumar00/days-death-to-a-banana")
        print("2. Click the 'Download' button")
        print("3. Extract the ZIP file")
        print("4. Organize files into data/images/ and data/labels.csv")


if __name__ == "__main__":
    main()
