# Vegetable Detector: Real-Time Transfer Learning Detector

An end-to-end deep learning and computer vision pipeline that classifies 15 types of vegetables in real-time. 

This project leverages an optimized Transfer Learning architecture to achieve **99.97% test accuracy**, deployed via a custom OpenCV interface with temporal smoothing for highly stable live-webcam inference.

## Model Architecture
The core of the classifier is a highly optimized Transfer Learning architecture. Rather than training from scratch, this pipeline leverages EfficientNetV2B0 as an advanced feature extractor and capped with a heavily regularized, custom-built classification head.

I've specifically chosen EfficientNetV2 for this project because it's a pretrained model where the base model was instantiated with weights pre-trained on the ImageNet dataset. During Phase 1 of training, these core layers were strictly frozen in order to protect model's foundational understanding of complex shapes, edges, and visual textures. Also, its parameter efficiency and real-time edge-device inference capabilities are off the roof & by utilizing Fused-MBConv blocks, EfficientNetV2 maximizes spatial feature extraction while minimizing computational latency, achieving high Frames Per Second (FPS) in the live webcam feed. 
**Data Pipeline:** 21,000 images across 15 classes, processed using TensorFlow's high-performance `tf.data.AUTOTUNE` pipeline for bottleneck-free loading.
The original 1000-class ImageNet top was removed (include_top=False), and a custom regularized head was attached specifically for the 15 vegetable classes

* **GlobalAveragePooling2D:** Replaces the traditional Flatten layer to heavily reduce the number of trainable parameters, making the model lighter and naturally resistant to overfitting.
* **Dense Layer 1:** Exactly 256 neurons, designed to find complex non-linear combinations of the features extracted by EfficientNet.
* **Dense Layer 2:** Exactly 128 neurons, continuing to distill the feature maps into highly specific vegetable profiles.
* **Regularization Matrix:** To prevent the model from memorizing the training data, Batch Normalization (standardizing activations to prevent internal covariate shift) and Dropout Layers (stochastically severing neural connections) are interspersed between the Dense layers.
* **Terminal Output Layer:** Exactly 15 neurons utilizing the SoftMax activation function to produce a normalized probability distribution across the 15 vegetable classes.

## Training Methodology
Training a transfer learning model on 21,000 images required strict data management and phased training approach to prevent destroying the highly valuable pre-trained weights.

* **Dataset Split & Ingestion** The model was trained on a strictly balanced, 21,000-image dataset (exactly 1,400 images per class), divided into three isolated partitions:
* **Training (70%):** 15,000 images dedicated to gradient calculation and weight updates.
* **Validation (15%):** 3,000 images utilized for active hyperparameter tuning, callback monitoring, and checkpointing.
* **Testing (15%):** 3,000 images held in absolute isolation, used exclusively for generating final real-world performance metrics.

* **Optimization Directives:** The compilation and training loop is mainly followed by strict mathematical directives to ensure convergence without overshooting the local minimums:
* Optimizer: Adam (Adaptive Moment Estimation), combining the best properties of AdaGrad and RMSProp.
* Loss Function: Categorical Cross-Entropy, paired perfectly with the one-hot encoded dataset.
* Batch Size: 32 (Calculated as per local hardware).
* Early Stopping: Monitored validation loss with a set patience threshold to halt training the moment the model began to overfit.
* ReduceLROnPlateau: Dynamically lowered the learning rate by factors of 0.2, allowing the optimizer to settle into microscopic local minimums.
* ModelCheckpoint: Silently monitored the epochs and permanently saved only the iteration with the highest validation accuracy, overwriting inferior versions.

## Key Features
* **High-Accuracy Transfer Learning:** Powered by a fine-tuned `EfficientNetV2B0` architecture, utilizing pre-trained ImageNet weights for highly efficient feature extraction.
* **Smart UI & Temporal Smoothing:** The OpenCV interface uses a 45-frame memory buffer (`collections.deque`) to "vote" on predictions, completely eliminating UI flickering caused by shadows or poor lighting.
* **Apple Silicon Optimized:** Bypassed standard prediction memory leaks to achieve ultra-smooth 30+ FPS on M-Series Mac hardware.


## Supported Classes
* **Classes:** Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato.
## Engineering Challenges Overcome
* **The "Sim-to-Real" Gap:** Realized that high dataset accuracy doesn't translate perfectly to webcams due to background noise and harsh lighting. Solved by implementing strict confidence thresholds (>85%) and a dynamic temporal voting system.
* **Dataset Bias:** Identified regional variance issues (e.g., Indian cucumbers vs. Western cucumbers) causing misclassifications with bottle gourds, requiring careful environmental testing & proper lighting conditions to detect it correctly


