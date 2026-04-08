# Vegetable Detector: Real-Time Transfer Learning Detector

An end-to-end deep learning and computer vision pipeline that classifies 15 types of vegetables in real-time. 

This project leverages an optimized Transfer Learning architecture to achieve **99.97% test accuracy**, deployed via a custom OpenCV interface with temporal smoothing for highly stable live-webcam inference.

## Key Features
* **High-Accuracy Transfer Learning:** Powered by a fine-tuned `EfficientNetV2B0` architecture, utilizing pre-trained ImageNet weights for highly efficient feature extraction.
* **Smart UI & Temporal Smoothing:** The OpenCV interface uses a 45-frame memory buffer (`collections.deque`) to "vote" on predictions, completely eliminating UI flickering caused by shadows or poor lighting.
* **Apple Silicon Optimized:** Bypassed standard prediction memory leaks to achieve ultra-smooth 30+ FPS on M-Series Mac hardware.

## Model Architecture & Methodology
1. **Data Pipeline:** 21,000 images across 15 classes, processed using TensorFlow's high-performance `tf.data.AUTOTUNE` pipeline for bottleneck-free loading.
2. **Phase 1 (Feature Extraction):** Attached a custom Global Average Pooling head and trained on frozen EfficientNetV2 base weights.
3. **Phase 2 (Fine-Tuning):** Unfroze the top 30 layers of the base model and applied a microscopic learning rate (`1e-5`) to specialize the core feature extractors for agricultural textures, pushing final test accuracy to **99.97%**.

## Supported Classes
* **Classes:** Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato.
## Engineering Challenges Overcome
* **The "Sim-to-Real" Gap:** Realized that high dataset accuracy doesn't translate perfectly to webcams due to background noise and harsh lighting. Solved by implementing strict confidence thresholds (>85%) and a dynamic temporal voting system.
* **Dataset Bias:** Identified regional variance issues (e.g., Indian cucumbers vs. Western cucumbers) causing misclassifications with bottle gourds, requiring careful environmental testing & proper lighting conditions to detect it correctly


