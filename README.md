# 🛰️ UC Merced Land Use Classification

Deep learning-based aerial scene classification using a custom CNN trained on the **UC Merced Land Use Dataset**. Includes a Streamlit web app for image classification with **Grad-CAM explainability**.

🚀 **Live Demo:** [aerialSeenClassification](https://aerialseenclassification-byy2tngvbgipol7zbxtdbr.streamlit.app/)

---

## 📌 Features

- Upload aerial/satellite images
- Predict land-use category
- Display prediction confidence
- Visualize model attention using Grad-CAM heatmaps

---

## 🗂️ Dataset

**UC Merced Land Use Dataset**
- 21 land-use classes
- 2,100 aerial images
- 256×256 RGB images

Example classes:
```
agricultural, airplane, beach, buildings, forest, freeway,
harbor, river, runway, parkinglot, storagetanks, tenniscourt
```

---

## 🧠 Model

A custom CNN trained for aerial image classification (`best_cnn.h5`).

The application uses **Grad-CAM** to highlight the regions of the image that most influenced the prediction.

---

## 🛠️ Technologies

- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Matplotlib

---

## 🚀 Run Locally

Clone the repository:
```bash
git clone https://github.com/TewodrosAdimas/aerialSeenClassification.git
cd aerialSeenClassification
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
aerialSeenClassification/
│
├── app.py
├── best_cnn.h5
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

**Tewodros Bewuket Adimas**
MSc Artificial Intelligence for Science and Technology
