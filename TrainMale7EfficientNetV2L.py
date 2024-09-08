import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import Dense, Conv2D, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# My bottom GPU is 1 and top is 0. I cannot get mirrored strategy to work with both without making training incredibly slow.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the Excel with the labels (filenames in column titled A and bone ages in column titled B)
labels = pd.read_excel('/mnt/d/AI_Datasets/boneage-training-dataset-male.xlsx')

# Split data to training and valdations sets 80:20 or what ever. I will use this method for different networks in the ensemble with different splits - the validation set will not always be the same. If refining one network I may need to define my own validation set and reduce its size (and increase the training dataset size) as required.

train_data, val_data = train_test_split(labels, test_size=0.25)

# Define the data generators
train_datagen = ImageDataGenerator()

# Set up Albumentations - code copied for each from Huginface demo. Tested and works nicely
def apply_augmentations(image):
    aug = A.Compose([
        A.HorizontalFlip(p=0.1),
        A.VerticalFlip(p=0.1),
        A.SafeRotate(
            limit=(-90, 90),  # ScaleFloatType
            interpolation=2,  # <class 'int'>
            border_mode=0,  # int
            value=None,  # ColorType | None
            mask_value=None,  # ColorType | None
            always_apply=None,  # bool | None
            p=0.15,  # float
        ),
        A.PixelDropout(
            dropout_prob=0.01,  # float
            per_channel=False,  # bool
            drop_value=0,  # ScaleFloatType | None
            mask_drop_value=None,  # ScaleFloatType | None
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.OpticalDistortion(
            distort_limit=(-0.15, 0.2),  # ScaleFloatType
            shift_limit=(-0.15, 0.2),  # ScaleFloatType
            interpolation=4,  # <class 'int'>
            border_mode=0,  # int
            value=None,  # ColorType | None
            mask_value=None,  # ColorType | None
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.ImageCompression(
            quality_lower=None,  # int | None
            quality_upper=None,  # int | None
            compression_type=0,  # ImageCompressionType
            quality_range=(70, 100),  # tuple[int, int]
            always_apply=None,  # bool | None
            p=0.05,  # float
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),  # ScaleFloatType
            contrast_limit=(-0.2, 0.2),  # ScaleFloatType
            brightness_by_max=True,  # bool
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.GaussianBlur(
            blur_limit=(3, 5),  # ScaleIntType
            sigma_limit=0,  # ScaleFloatType
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0),  # ScaleFloatType
            mean=0,  # float
            per_channel=True,  # bool
            noise_scale_factor=1,  # float
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.Defocus(
            radius=(1, 2),  # ScaleIntType
            alias_blur=(0.1, 0.5),  # ScaleFloatType
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.Blur(
            blur_limit=3,  # ScaleIntType
            p=0.05,  # float
            always_apply=None,  # bool | None
        ),
        A.UnsharpMask(
            blur_limit=(3, 7),  # ScaleIntType
            sigma_limit=0.0,  # ScaleFloatType
            alpha=(0.2, 0.5),  # ScaleFloatType
            threshold=10,  # int
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.Sharpen(
            alpha=(0.2, 0.5),  # tuple[float, float]
            lightness=(0.5, 1.0),  # tuple[float, float]
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.ISONoise(
            color_shift=(0.01, 0.05),  # tuple[float, float]
            intensity=(0.1, 0.5),  # tuple[float, float]
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.Perspective(
            scale=(0.025, 0.025),  # ScaleFloatType
            keep_size=True,  # bool
            pad_mode=0,  # int
            pad_val=0,  # ColorType
            mask_pad_val=0,  # ColorType
            fit_output=True,  # bool
            interpolation=4,  # <class 'int'>
            always_apply=None,  # bool | None
            p=0.1,  # float
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),  # ScaleIntType
            always_apply=None,  # bool | None
            p=0.1,  # float
        )
    ])
    augmented = aug(image=image)["image"]
    return augmented

# Function to call the augmentation function and use EfficientNet's or any other network's built in preprocessing. This is sometimes not needed, in which case it acts as a passthrough. Sometimes it is needed, so it is safest to do it for all.
def preprocess_image(image):
    # Apply augmentations using Albumentations
    augmented = apply_augmentations(image)
    # Preprocess using EfficientNet's preprocess_input
    preprocessed = preprocess_input(augmented)
    return preprocessed
    
# For EfficientNet, the batch size has to be small otherwise I get an out of memory error. With ResNet I can go up to 32.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='/mnt/d/AI_Datasets/boneage-training-dataset/male_training',
    x_col="A",
    y_col="B",
    target_size=(480, 480),
    color_mode='rgb',
    class_mode='raw',
    batch_size=16,
    shuffle=True,
    preprocessing_function=preprocess_image
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory='/mnt/d/AI_Datasets/boneage-training-dataset/male_training',
    x_col="A",
    y_col="B",
    target_size=(480, 480),
    color_mode='rgb',
    class_mode='raw',
    batch_size=16,
    shuffle=False,
    preprocessing_function=preprocess_image
)

# Define model and include weights. I seem to need the weight for the larger models, but it limits me to the input size used. ResNet does well training from scratch, but EfficientNet doesn't seem to.
base_model = EfficientNetV2L(weights="imagenet", include_top=False)
x = base_model.output

# Freeze the base model layers - this is not needed as I am not training these layers anyway, but maybe needed if I am using some other pre-trained weights?
# base_model.trainable = False

# Load trained weights
# model.load_weights('/mnt/d/AI_Datasets/models/mae_79_6.07_88.50.weights.h5')

# Add custom layers on top
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Add a dropout layer to reduce overfitting - seems to work very well.
x = BatchNormalization()(x)  # Add batch normalization layer
output_tensor = Dense(1, activation='linear')(x)
model = Model(inputs=base_model.input,  outputs=output_tensor)

# I have given up with gradient accumulation, a small batch size may be better to avoid overfitting anyway.

# Define optimizer and set the initial learning rate
# optimizer = Adam(learning_rate=0.001)

# Compile model with the optimizer above
model.compile(optimizer=Adam(), loss='mse', metrics=['mae']) # I could put optimizer=Adam() instead of optimizer=optimizer and not specify the learning rate. Adam keeps it an 0.001 to start anyway.
model.summary()

# Define the callbacks
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=60, min_lr=0.0001)

# This is a bit annoying. If I just save validation loss and mae best models, I miss one that is slightly less good on those metrics, but much better on the training set. This code saves multiple duplicate models when several metrics improve, which is a waste, but ensures I don't miss anything. I then have to delete duplicate models.
checkpoint_valloss = callbacks.ModelCheckpoint(
    '/mnt/d/AI_Datasets/models/valloss_{epoch:02d}_{val_mae:.2f}_{val_loss:.2f}.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

checkpoint_valmae = callbacks.ModelCheckpoint(
    '/mnt/d/AI_Datasets/models/valmae_{epoch:02d}_{val_mae:.2f}_{val_loss:.2f}.keras',
    monitor='val_mae',
    verbose=1,
    save_best_only=True,
    mode='min'
)
checkpoint_loss = callbacks.ModelCheckpoint(
    '/mnt/d/AI_Datasets/models/loss_{epoch:02d}_{mae:.2f}_{loss:.2f}.keras',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

checkpoint_mae = callbacks.ModelCheckpoint(
    '/mnt/d/AI_Datasets/models/mae_{epoch:02d}_{mae:.2f}_{loss:.2f}.keras',
    monitor='mae',
    verbose=1,
    save_best_only=True,
    mode='min'
)

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=180)

# Save to Excel every epoch - I chart the progress in Excel by copying this file as it is training and inserting a graph.
class SaveTrainingStatsCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the training statistics
        training_stats = {
            'Epoch': epoch + 1,
            'Loss': logs['loss'],
            'MAE': logs['mae'],
            'Validation Loss': logs['val_loss'],
            'Validation MAE': logs['val_mae']
        }

        # Create a new DataFrame with the training stats
        new_row = pd.DataFrame([training_stats])

        # Load the Excel file (if it exists)
        excel_file = '/mnt/d/AI_Datasets/training_stats_live.xlsx'
        if os.path.exists(excel_file):
            existing_df = pd.read_excel(excel_file, sheet_name='Sheet1')
        else:
            existing_df = pd.DataFrame()  # Create an empty DataFrame

        # Append the new row to the existing DataFrame
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)

        # Save the updated DataFrame back to the Excel file
        updated_df.to_excel(excel_file, sheet_name='Sheet1', index=False)


# Train with mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1000,
    callbacks=[checkpoint_loss, checkpoint_mae, checkpoint_valloss, checkpoint_valmae, early_stopping, reduce_lr, SaveTrainingStatsCallback()]
)

# To run this, copy this file and rename it to TrainMale7.py. Put it in the AI_datasets directory, then from WSL: Python3 /mnt/d/AI_Datasets/TrainMale7.py
# To stop the script Ctrl Z and then type kill %n where n is the number shown after you have pressed Ctrl Z
