# Banana Ripeness Detection Project 🍌

This project uses Machine Learning (XGBoost) to predict the ripeness of bananas based on their images.

## 🚀 How to Run the Project

Follow these steps in order to get the project running.

### 1️⃣ Install Dependencies
First, make sure you have all the required Python libraries installed.
```bash
pip install -r requirements.txt
```

### 2️⃣ Get the Data
You need the banana image dataset.
*   **Automatic:** Run the downloader script (requires Kaggle API setup).
    ```bash
    python download_dataset.py
    ```
*   **Manual:**
    1.  Download from [Kaggle](https://www.kaggle.com/datasets/anishkumar00/days-death-to-a-banana).
    2.  Extract the zip.
    3.  Ensure your folder structure looks like this:
        ```
        portfolio_site/
        └── Data/
            ├── images/       # Contains all banana images
            └── labels.csv    # Contains the label data
        ```

### 3️⃣ Train the Model
Once the data is ready, train the machine learning model.
```bash
python train_model.py
```
*   This script analyzes the images, extracts features, trains the XGBoost model, and saves the model to `banana_xgboost_model.json`.

### 4️⃣ Run the Web App
Start the Streamlit interface to use the model.
```bash
streamlit run app.py
```
*   The app will open in your browser (usually at `http://localhost:8501`).
*   Upload a banana image to see the ripeness prediction and suggestions.

---

## 📂 Project Structure & Files

Here is a quick guide to what each file does:

| File Name | Description |
| :--- | :--- |
| **`app.py`** | The main **Streamlit Web Application**. Handles the UI, image upload, and connects to the ML model for predictions. |
| **`train_model.py`** | The **Machine Learning Engine**. Reads images, extracts features (color, texture), trains the XGBoost model, and saves it. |
| **`download_dataset.py`** | Helper script to automatically download the dataset from Kaggle if you have the API configured. |
| **`run_demo.py`** | A diagnostic script to check if your environment is set up correctly and dependencies are installed. |
| **`requirements.txt`** | Lists all the Python libraries required to run this project. |
| **`banana_xgboost_model.json`** | The saved, trained model file (generated after running `train_model.py`). |
