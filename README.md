PestVision AI – Eco-Smart Pest Detection using Deep Learning

PestVision AI is an intelligent image-based pest detection system designed to assist farmers and agricultural researchers in identifying pest species accurately and efficiently. Built using Convolutional Neural Networks (CNN) and powered by the MobileNetV2 architecture, this project combines the efficiency of modern deep learning models with a user-friendly interface to deliver real-time pest classification.

The system analyzes uploaded pest images and predicts the pest type with high accuracy while providing Grad-CAM visualizations to show which regions influenced the model’s decision. The web interface, developed with Streamlit, is designed with a clean green-white theme to represent sustainability and nature. The app ensures accessibility even for users without technical expertise, making pest identification faster, smarter, and more sustainable.

Key Features:
• Smart pest image classification using CNN (MobileNetV2 backbone)
• Interactive web app built with Streamlit for real-time predictions
• Grad-CAM visualization for interpretability and model transparency
• Modern, minimalist interface with green-white eco-inspired design
• Compatible with TensorFlow and Keras deep learning frameworks
• Supports image formats like JPG, JPEG, and PNG
• Fully local and customizable for deployment

Technologies Used:
Python, TensorFlow, Keras, NumPy, Pandas, OpenCV, Streamlit, Matplotlib, JSON

Model Training and Deployment:
The deep learning model was trained on pest image datasets with separate training, testing, and validation splits to ensure balanced accuracy. The trained Keras model (.keras) and class mapping JSON file are integrated into the Streamlit application for seamless local execution.

PestVision AI stands as a step toward sustainable agriculture, helping in early pest detection and efficient crop protection through the power of deep learning and computer vision.