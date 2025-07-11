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
    Added "split" argument (train/val/test) to support different dataset splits 
    without creating separate classes.
    '''
    def __init__(self, *args, **kwargs):
        # Extract split keyword (default: 'train') before calling parent constructor
        self.split = kwargs.pop('split', 'train')  # train | val | test

        # call parent constructor with remaining kwargs
        super().__init__(*args, **kwargs)

    def _scan(self):
        # Dataset is expected to be located under
        #   ./dataset/prep/SatelliteImages_s512_o128/<split>/CL/*.png
        # where <split> is train, val or test

        self.dataset_path = os.path.join(self.dataset_dir, 'prep', 'SatelliteImages_s512_o128', self.split)
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path

        clean_path = os.path.join(self.dataset_path, 'CL')
        if os.path.exists(clean_path):
            # recursively gather all .png files under CL directory
            self.img_paths = [f for f in os.listdir(clean_path) if f.lower().endswith('.png')]
        else:
            raise RuntimeError(f'Cannot find CL directory for split "{self.split}" at {clean_path}')

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        clean_img_path = os.path.join(self.dataset_path, 'CL', file_name)
        clean_img = self._load_img(clean_img_path)

        return {'clean': clean_img}