# BoneAgeTraining
Bone age training using the RSNA dataset.
This trains using Keras in WSL2. I would like to try PyTorch, but haven't managed to figure out the workflow. I think there may be more flexibility with PyTorch and imagenet22k weights, also, it'll work in Windows.

EfficientNetV2L trains to an MAE of 4.35 on the test dataset. I have improved this very slightly by using a ResNet model and weighting the result 17:1 towards the EfficientNet model)

![AI model performance](https://github.com/user-attachments/assets/c9b180fc-9b0a-4ee7-bfce-78c8fb52f2ab)

Things that work well:

Albumentations
Mixed precision training seems to work.
Batch normalisation
Dropout
EfficientNet with imagenet weights - much slower to train, uses more memory, but better MAE
ResNet without pretrained weights - much faster to train and uses less memory.

Things I cannot get to work:
Using two GPUs
Gradient accumulation
Saving only one model per epoch
