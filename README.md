# AI_commentary_application_for_sports

We aim at developing such system capable of action tracking and understanding in basketball games using computer vision approaches and ideas alongside deep learning models such as Detectron2.
Classification of 🏀 Basketball Actions using 3D-CNN Models trained on the SpaceJam Dataset.

## Dataset
SpaceJam Basketball Action Dataset was used to train the R(2+1)D CNN model for video/action classification of basketball actions. This dataset contains two datasets (clips->.mp4 files and joints -> .npy files) of basketball single-player actions. The size of the two final annotated datasets is about 32,560 examples.

## Demo
https://github.com/wooing1084/AI_commentary_application_for_sports/assets/97783148/d2464c0f-80d8-4762-937f-0e88864514f5


## Dependencies
- Python == 3.10.12
- OpenCV == 4.8.0
- Detectron2 
- Cuda == 11.8
- Pytorch-Cuda == 2.1.1
- Numpy == 1.23.5
- Matplotlib == 3.7.1

## reference
- https://github.com/Basket-Analytics/BasketTracking
- https://github.com/hkair/Basketball-Action-Recognition
- https://github.com/simonefrancia/SpaceJam
