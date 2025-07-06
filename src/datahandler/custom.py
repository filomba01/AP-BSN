import os

import h5py

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class SatelliteImages(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        # Use the standard dataset directory structure
        dataset_path = os.path.join(self.dataset_dir, 'SatelliteImages')
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        # Scan for PNG files in the dataset directory
        self.img_paths = []
        for file_name in os.listdir(dataset_path):
            if file_name.lower().endswith('.png'):
                self.img_paths.append(file_name)
        
        print(f"Found {len(self.img_paths)} PNG images in {dataset_path}")
        for img_path in self.img_paths:
            print(f"  - {img_path}")

    def _load_data(self, data_idx):
        # Load clean satellite images (no paired noisy images available)
        file_name = self.img_paths[data_idx]
        dataset_path = os.path.join(self.dataset_dir, 'SatelliteImages')
        
        clean_img = self._load_img(os.path.join(dataset_path, file_name))

        return {'clean': clean_img}  # only clean image dataset


@regist_dataset
class prep_SatelliteImages(DenoiseDataSet):
    '''
    dataset class for prepared satellite images which are cropped with overlap.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'prep/SatelliteImages_s512_o128')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        
        clean_path = os.path.join(self.dataset_path, 'CL')
        if os.path.exists(clean_path):
            for root, _, files in os.walk(clean_path):
                self.img_paths = [f for f in files if f.lower().endswith('.png')]

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        clean_img = self._load_img(os.path.join(self.dataset_path, 'CL', file_name))

        return {'clean': clean_img}