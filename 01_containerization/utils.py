import numpy as np
from tensorflow import keras
import cv2
import os

IMG_PATH = os.environ['DATA_DIR']

class DataGenerator(keras.utils.Sequence):
    def __init__(self, patient_id, target_class, batch_size=32, size=(512, 512), seed=1, shuffle=False, **kwargs):
        """
        Custom data generator for segmentation tasks.
        
        Args:
        - img_dir: Directory containing the input images
        - mask_dir: Directory containing the corresponding masks
        - batch_size: Number of samples per batch
        - size: The target size for resizing the images and masks
        - seed: Random seed for reproducibility
        - shuffle: Whether to shuffle the dataset after each epoch
        """
        super().__init__(**kwargs)
        
        # List image and mask files
        self.patient_id = patient_id
        self.target_class = target_class
        
        self.batch_size = batch_size
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        
        # Ensure the number of images matches the number of masks
        assert len(self.patient_id) == len(self.target_class), \
            "The number of images and masks must be the same"
        
        self.indexes = np.arange(len(self.patient_id))  # Indices for shuffling
        
        # If shuffle is enabled, shuffle the indices after each epoch
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.patient_id) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates a batch of data (images and corresponding masks).
        
        Args:
        - index: The index of the batch.
        
        Returns:
        - A batch of images and masks
        """
        # Get batch indices
        batch_indices = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # Initialize empty arrays for the batch
        images = []  
        target = []
        
        for i, idx in enumerate(batch_indices):
            # Load and preprocess image
            img = cv2.imread(IMG_PATH+self.patient_id[idx]+'.png',1)  # Read image
            img = cv2.resize(img, self.size)  # Resize to target size
            img = img / 255.0  # Normalize to [0, 1]
            
            # Add image and mask to the batch arrays
            images.append(img)
            target.append(self.target_class[idx])
        
        return np.array(images), np.array(target)

    def on_epoch_end(self):
        """
        Shuffle the dataset after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)