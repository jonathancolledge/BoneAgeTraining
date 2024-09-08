# BoneAgeTraining
Bone age training using the RSNA dataset
This trains using Keras. I would like to try PyTorch, but haven't managed to figure out the workflow.

EfficientNetV2L trains to an MAE of 4.35 on the test dataset. I have improved this very slightly by using a ResNet model and weighting the result 17:1 towards the EfficientNet model)

![AI model performance](https://github.com/user-attachments/assets/b1b83ad8-edf4-4929-b4f3-b5a225939047)
