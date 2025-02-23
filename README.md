Background_Removal
This project implements background removal using U²-Net, a deep learning model designed for salient object detection. U²-Net efficiently segments the foreground object from the background, enabling high-quality background removal in images. The implementation is done in Python, utilizing PyTorch for model inference.

Requirements
Before running the application locally, ensure the following are installed on your system:

Download the U²-Net model file (u2net_portrait.pth) from one of the following sources:

Google Drive 
https://drive.google.com/file/d/1IG3HdpcRiDoWNookbncQjeaPN28t90yW/view
Baidu Pan (Extraction Code: chgd)
After downloading, place the file in the directory: ./saved_models/u2net_portrait/.

Python 3: Recommended version 3.x. Download instructions can be found here.

Pip: Python package installer (should come with Python).

Install the following Python packages (preferably in a virtual environment):

numpy==1.15.2
scikit-image==0.14.0
torch
torchvision
pillow==8.1.1
opencv-python
paddlepaddle
paddlehub
gradio
