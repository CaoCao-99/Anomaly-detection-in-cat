Cat Anomaly Behavior Detection Project
Overview
This project aims to detect abnormal behaviors in cats to monitor their health. The methodology involves using the Yolov8 model for keypoint estimation, followed by anomaly detection techniques to identify irregular behaviors.

Project Objective
The primary goal of this research is to detect health abnormalities in cats by identifying unusual behaviors.

Methodology
Input Video: The process starts with a video of a cat.
Keypoint Estimation:
The video is fed into a pre-trained keypoint estimation model (Yolov8).
The model extracts keypoints representing various parts of the cat's body.
Keypoint Distance Calculation:
Calculate the distances between keypoints.
Normalization with FFT:
Apply Fast Fourier Transform (FFT) to the distances.
Normalize the FFT output.
Anomaly Detection:
Use an LSTM autoencoder to learn from normalized keypoint distances of normal behavior.
Reconstruct the distances and calculate the difference between the original and reconstructed values to detect anomalies.
Framework
The framework of the model is outlined below:

Input: Video of a cat.
Keypoint Estimation (Yolov8):
Extract keypoints from the video.
Distance Calculation:
Compute the differences between each keypoint.
FFT Normalization:
Apply FFT to the differences.
Normalize the FFT results.
LSTM Autoencoder:
Feed normalized keypoint distances into an LSTM autoencoder.
Reconstruct the normalized distances.
Calculate the difference between original and reconstructed distances.
Anomaly Detection:
Significant differences indicate potential abnormal behavior.
Repository Structure
data/: Contains the video files and any related datasets.
models/: Pre-trained models and model weights.
scripts/: Scripts for keypoint estimation, distance calculation, FFT normalization, and anomaly detection.
notebooks/: Jupyter notebooks for exploratory data analysis and model training.
results/: Outputs and results from model predictions and evaluations.
Installation
To set up the project locally, follow these steps:

Clone the repository:
git clone https://github.com/yourusername/cat-anomaly-behavior-detection.git
cd cat-anomaly-behavior-detection
Install the required dependencies:

pip install -r requirements.txt

Usage
Preprocess the Video:
Use the scripts in scripts/preprocess_video.py to convert video files into a suitable format for keypoint estimation.
Run Keypoint Estimation:
Use the script scripts/keypoint_estimation.py to extract keypoints from the video.
Calculate Distances and Normalize:
Run scripts/calculate_distances.py to compute the distances between keypoints and apply FFT normalization.
Anomaly Detection:
Use scripts/anomaly_detection.py to run the LSTM autoencoder and detect anomalies.
Contributing
If you would like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Special thanks to the developers of Yolov8 for providing an excellent keypoint estimation model.
Thanks to all contributors for their efforts in improving this project.
For further information, please contact [yourname@example.com].
