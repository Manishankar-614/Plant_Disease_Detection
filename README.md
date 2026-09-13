# Plant Disease Multi-Task Classifier

This project trains a MobileNetV2-based multi-task model to classify both the plant part (Fruit, Leaf, Stem) and its disease.

## 🚀 Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
    cd your-project-name
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    *(First, create a requirements.txt file: `pip freeze > requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset**
    * Download the `dataset_split.zip` file from Google Drive:
    * [Download dataset_split.zip](https://drive.google.com/file/d/1xsvLHa7FBgwFQ9ZaIH2YuHuvw_5EowaV/view?usp=sharing)
    * Unzip the file inside the project folder. Your project directory should now look like this:
        ```
        your-project-name/
            ├── dataset_split/
            ├── train_mobilenet.py
            ├── predict_mobilenet.py
            ├── README.md
            └── .gitignore
        ```

## 🖥️ Usage

**1. To Train the Model:**
```bash
python train_mobilenet.py
```

**2. To Run Predictions:**
This will first evaluate on the 'test' set, then open an interactive window to select your own images.
```bash
python predict_mobilenet.py
```
