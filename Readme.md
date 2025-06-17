# 🧠 Alzheimer's Disease Detection using PET Scans

This project focuses on detecting and classifying different stages of Alzheimer's Disease using PET (Positron Emission Tomography) scans with the help of deep learning models in MATLAB.

## 🎯 Objective

To accurately classify patients into five stages:

- CN – Cognitively Normal  
- MCI – Mild Cognitive Impairment  
- EMCI – Early MCI  
- LMCI – Late MCI  
- AD – Alzheimer's Disease

## 🛠️ Tools & Technologies

- MATLAB (Deep Learning Toolbox)
- Image Processing Toolbox
- PET scan datasets (DICOM, PNG, or JPG)

## 🧠 Model Overview

The model uses a combination of:

- **CNN (Convolutional Neural Network)** for image feature extraction  
- **LSTM (Long Short-Term Memory)** for temporal/spatial learning  
- **Ensemble Learning** to boost classification performance

## 🧪 Dataset

- Medical PET scan images sourced from publicly available datasets  
- Preprocessing steps include image resizing, normalization, and augmentation

## 📊 Results

- Achieved high accuracy in distinguishing between different AD stages  
- Evaluated using:
  - Confusion matrix  
  - ROC curve  
  - Accuracy and loss plots

## 🚀 How to Run

1. Open the `.m` files and `.mlx` live scripts in MATLAB.  
2. Ensure the dataset path is correctly set.  
3. Run the main script to train and evaluate the model.

## Output

Figures/output1.jpg
Figures/output2.jpg

## 📄 License

This project is released under the MIT License.

---

Feel free to use this work for academic or research purposes. Contributions are welcome!
