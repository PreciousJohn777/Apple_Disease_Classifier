# Apple Disease Classification Using CNN

## CE1 Mini Project

This project uses a Convolutional Neural Network (CNN) to classify apple images into two categories:

- Rotten Apple
- Formalin-mixed Apple

## Dataset

The dataset contains:

- Formalin-mixed Apple: 643 images
- Rotten Apple: 630 images

Total Images: 1,273

## Technologies Used

- Python
- TensorFlow/Keras
- Streamlit
- NumPy
- Matplotlib

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
streamlit run app.py
```

3. Upload an apple image.

4. The application predicts whether the apple is:

- Rotten Apple
- Formalin-mixed Apple

## Project Structure

```
Apple_Disease_Classifier/
│
├── app.py
├── train_model.py
├── train_model.ipynb
├── models/
│   └── apple_classifier.keras
├── dataset/
├── requirements.txt
├── README.md
└── .gitignore
```

## Author

Precious John