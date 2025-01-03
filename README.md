Overview
This repository contains code for a deep learning framework designed to classify medical implants (shoulder, hip, and knee) from X-ray images using MobileNetV2, NASNetMobile, Self-Attention, 
and SEBlock (Squeeze-and-Excitation Block). The pipeline includes feature extraction, hierarchical fusion, and ensemble learning with majority voting. Visualization tools such as Grad-CAM and
t-SNE are incorporated for interpretability.


Setup
Install Required Libraries:

Ensure you have the following Python libraries installed:
pip install tensorflow keras scikit-learn matplotlib seaborn xgboost


Prepare Data:
Organize your data directories for training and testing:

Shoulder_dataset/
  train/
    class1/
    class2/
  test/
    class1/
    class2/
Hip_dataset/  # Similar structure
Knee_dataset/ # Similar structure


Pretrained Weights:

Download the pretrained models and save them in the weights/ directory. Ensure filenames match those referenced in the code (e.g., dataset1_mobilenet.h5, dataset1_nasnet.h5).
Directory Structure:

Verify the directory structure for outputs (e.g., Grad-CAM/results/).
Understanding the Code


Preprocessing:
Rescales images (1./255) and applies augmentation techniques.
Ensures images are resized to 224x224.
Feature Extraction with Pretrained Models:

MobileNetV2 and NASNetMobile serve as backbones.
Models are enhanced with Self-Attention and SEBlock.
Outputs from both models are fused using an element-wise addition.
Fusion Techniques:

Task-Based Fusion: Horizontal concatenation of features from the same implant type.
Global Fusion: Combines features across all implant types using Independent Component Analysis (ICA).
Visualization:

t-SNE visualizes feature separability for interpretability.
Grad-CAM generates heatmaps highlighting critical image regions contributing to predictions.
Ensemble Learning:

Implements soft-voting classifiers with Logistic Regression, SVM, KNN, XGBoost, and Naive Bayes.
Majority voting determines the final class predictions.
Evaluation Metrics:

Confusion matrix, accuracy, recall, precision, F1 score, and AUC.



How to Run
Feature Extraction:

Train or load pretrained MobileNetV2 and NASNetMobile models using the provided SEBlock and Self-Attention.
Classification:

Extract features for shoulder, hip, and knee datasets.
Perform hierarchical fusion and train ensemble classifiers.
Visualization:

Run t-SNE and Grad-CAM scripts to visualize feature separability and explain predictions.
Modifications for New Datasets
Add New Classes:

Update class_names to include additional classes.
Ensure datasets for the new classes follow the same directory structure.
Additional Modalities:

For additional modalities (e.g., CT or MRI), preprocess data and include in the fusion process.
Hyperparameter Tuning:

Adjust learning rates, batch sizes, and the number of epochs in the training scripts for different dataset sizes.
Integration:

Add new datasets to the hierarchical fusion pipeline by modifying the respective feature extraction and fusion scripts.
Important Notes
Scalability:

The current setup is designed for three implant types. Adding more tasks requires careful tuning of the model layers and fusion mechanisms.
Performance:

The code uses ensemble classifiers and visualization tools, which may require high computational resources for large datasets.
Interpretability:

Grad-CAM and t-SNE enhance interpretability but do not directly contribute to the classification performance.
