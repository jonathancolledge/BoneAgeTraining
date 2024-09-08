# BoneAgeTraining
Bone age training using the RSNA dataset.
This trains using Keras in WSL2. I would like to try PyTorch, but haven't managed to figure out the workflow. I think there may be more flexibility with PyTorch and imagenet22k weights, also, it'll work in Windows.

EfficientNetV2L trains to an MAE of 4.35 on the test dataset. I have improved this very slightly by using a ResNet model and weighting the result 17:1 towards the EfficientNet model)

![AI model performance](https://github.com/user-attachments/assets/c9b180fc-9b0a-4ee7-bfce-78c8fb52f2ab)

Things that work well:<br><br>

Albumentations<br>
Mixed precision training seems to work.<br>
Batch normalisation<br>
Dropout<br>
EfficientNet with imagenet weights - much slower to train, uses more memory, but better MAE<br>
ResNet without pretrained weights - much faster to train and uses less memory.<br>
ReduceLRonPlateau - I have to tweak the patience according to which network I am training.
Early stopping - again, I have to tweak the patience - it is hard not to set it too high!

Things I cannot get to work:<br><br>
Using two GPUs<br>
Gradient accumulation<br>
Saving only one model per epoch<br>
