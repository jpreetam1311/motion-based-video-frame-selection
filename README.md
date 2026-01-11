# 🚦 Data-Efficient Traffic Accident Video Representation Using Dense Optical Flow & Clustering

## 📌 Overview
Training deep learning models on raw traffic videos is computationally expensive and inefficient due to extreme temporal redundancy. Most consecutive frames contain little new information, especially before and after key events such as traffic collisions.

This project proposes a **data-efficient video preprocessing pipeline** that identifies and retains **high-information frames** from traffic accident videos by leveraging **dense optical flow** and **unsupervised clustering**. The goal is to reduce redundant frames while preserving motion patterns that are critical for accident detection and analysis.

---

## 🎯 Motivation
Traffic accident datasets typically suffer from:

- Large video sizes with minimal frame-to-frame variation  
- Overrepresentation of static or low-motion frames  
- Increased training time and risk of overfitting when using all frames  

Rather than feeding every frame into a deep learning model, this project focuses on **motion-centric frame selection**, ensuring that only representative and meaningful frames are retained for downstream training.

---

## 🧠 Key Idea
Traffic collisions exhibit a distinctive **motion pattern**:

- Vehicles appear as **dynamic objects** with strong motion signals before impact  
- Immediately after collision, motion rapidly decreases as vehicles become static  
- This **motion-to-static transition** is a strong indicator of an accident event  

Dense optical flow captures this behavior at the pixel level, making it a suitable representation for identifying informative frames.

---

## 🛠️ Methodology

### 1. Video Frame Sampling
- Traffic accident videos (`.mp4`) are read using OpenCV  
- Frames are temporally downsampled by skipping intermediate frames to reduce redundancy  

---

### 2. Dense Optical Flow Computation
- Dense optical flow is computed between consecutive grayscale frames using the **Farneback algorithm**
- For each pixel, motion is represented as:
  - **Magnitude** (speed of motion)
  - **Angle** (direction of motion)

Static regions naturally produce near-zero flow, while moving objects generate strong motion vectors.

---

### 3. Motion Feature Representation
Instead of storing full optical flow maps, each frame transition is summarized using:

- Histogram of flow magnitudes (16 bins)  
- Histogram of flow angles (16 bins)  

These histograms are concatenated into a compact **32-dimensional motion descriptor**, capturing the global motion pattern of the frame pair.

This step dramatically reduces dimensionality while preserving motion information.

---

### 4. Unsupervised Clustering (K-Means)
- Motion descriptors from each video are clustered using **K-Means**
- Each cluster represents a distinct motion pattern (e.g., steady motion, sudden stop, collision)

---

### 5. Representative Frame Selection
- For each cluster, the frame closest to the cluster centroid is selected  
- These frames act as **key frames** that summarize the video’s motion dynamics  

This ensures the final dataset contains **diverse yet non-redundant frames**.

---

## 📈 Benefits
- Significant reduction in the number of training frames  
- Lower computational and memory cost for deep learning models  
- Preserves accident-specific motion cues  
- Improves data efficiency without manual labeling  

---

## 🧪 Example Use Case
This preprocessing pipeline can be used to:

- Prepare training data for accident detection or prediction models  
- Improve efficiency of CNNs or video-based deep learning architectures  
- Analyze motion patterns in traffic surveillance footage  

---

## 🧰 Tech Stack
- **Python**
- **OpenCV**
- **NumPy**
- **scikit-learn**
- **Matplotlib**

---

