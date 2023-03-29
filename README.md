# XarpAi Lung Opacity Detector
An Ai powered desktop app that auto detects opacities on chest x-rays.

Opacities are characteristic signs of TB and Pneumonia. This app is a prototype high volume diagnosis support tool. It auto analyzes chest x-rays and highlights potential opacities. Radiologists can then review the highlighted areas and use their clinical judgment to make a final diagnosis.

The app ouputs an interactive image. Clicking on the image causes the bounding boxes to appear and disappear. This feature plus image zoom makes it easier for a person to review the output image. This a flask app running on the desktop. Internally the app is powered by a Faster-RCNN model that was trained on data from four chest x-ray detection datasets.
These included:
- The TBX11K Simplified Tuberculosis dataset
- The Kaggle RSNA Pneumonia Detection Challenge
- The Kaggle VinBigData Chest X-ray Abnormalities Detection competition
- The Kaggle SIIM-FISABIO-RSNA COVID-19 Detection competition

<br>
<img src="https://github.com/vbookshelf/XarpAi-Lung-Opacity-Detector/blob/main/images/tb0688.png" height="400"></img>
<i>Sample prediction<br>This opacity is an indicator of TB</i><br>
<br>

To verify that the model could generalize I validated it on out of sample data i.e. datasets from sources that the model had not seen to during training. These included:
- The Shenzhen and Montgomery Tuberculosis datasets
- The DA and DB Tuberculosis datasets
- The Child Chest X-Ray Images Pneumonia dataset
