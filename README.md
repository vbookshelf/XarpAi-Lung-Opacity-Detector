# XarpAi Lung Opacity Detector
An Ai powered desktop app that auto detects opacities on chest x-rays.

Opacities are characteristic signs of TB and Pneumonia. This app is a prototype high volume diagnosis support tool. It auto analyzes chest x-rays and highlights potential opacities. Radiologists can then review the highlighted areas and use their clinical judgment to make a final diagnosis.

The app ouputs an interactive image. Clicking on the image causes the bounding boxes to appear and disappear. This feature plus image zoom makes it easier for a person to review the output image. This a flask app running on the desktop.

<br>
<img src="https://github.com/vbookshelf/XarpAi-Lung-Opacity-Detector/blob/main/images/tb0688.png" height="400"></img>
<i>Sample prediction<br>This opacity is an indicator of TB</i><br>
<br>

## Demo

<br>

## 1- Main Features

- Runs on the CPU. 
- Images are interactive. Clicking an image causes the bounding boxes to disappear.
- A user can zoom into an image by using the desktop zoom feature that’s built into Mac and Windows.
- Multiple images can be submitted
- Free to use. Free to deploy. No server rental costs like with a web app.
- Runs locally without needing an internet connection

<br>

## 2- Cons

- It’s not a one click setup. The user needs to have a basic knowledge of how to use the command line to set up a virtual environment, download requirements and launch a python app.
- The model’s ability to generalize to real world data is unproven.
- The model may predict multiple overlapping bounding boxes. I've not fixed this because this is a prototype and the bounding boxes can be hidden by simply clicking on the image.

<br>

## 3- Training and Validation

Internally the app is powered by a Faster-RCNN model that was trained on data from four chest x-ray detection datasets.
These included:
- The TBX11K Tuberculosis dataset
- The Kaggle RSNA Pneumonia Detection Challenge
- The Kaggle VinBigData Chest X-ray Abnormalities Detection competition
- The Kaggle SIIM-FISABIO-RSNA COVID-19 Detection competition

To verify that the model could generalize I validated it on out of sample data i.e. datasets from sources that the model had not seen to during training. These included:
- The Shenzhen and Montgomery Tuberculosis datasets
- The DA and DB Tuberculosis datasets
- The Child Chest X-Ray Images Pneumonia dataset

The app displays opacity bounding boxes, but internally the model is also trained to predict a bounding box for the lungs. Therefore, if the lungs are not detected then the app outputs an error message to say that the image was not a chest x-ray.
