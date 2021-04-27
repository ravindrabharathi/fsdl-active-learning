"""
W&B drought watch benchmark data set
"""
from pathlib import Path
import os
import shutil
import zipfile

from torchvision import transforms
import h5py
import toml

from text_recognizer.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset

import tensorflow as tf
import random
import numpy as np

IMG_DIM = 65
NUM_CLASSES = 4
N_TRAIN = 20000
N_VAL = 10000
BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "droughtwatch"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "droughtwatch"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "droughtwatch"
PROCESSED_DATA_FILE_TRAINVAL = PROCESSED_DATA_DIRNAME / "trainval.h5"
PROCESSED_DATA_FILE_POOL = PROCESSED_DATA_DIRNAME / "pool.h5"


class DroughtWatch(BaseDataModule):
    """
    DataModule for W&B drought watch benchmark data set. Downloads data as ZIP in TFRecord format and converts it to PyTorch format.
    Learn more at https://wandb.ai/wandb/droughtwatch/benchmark or https://github.com/wandb/droughtwatch.
    """

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--n_train_images", type=int, default=N_TRAIN)
        parser.add_argument("--n_validation_images", type=int, default=N_VAL)
        parser.add_argument("--bands", type=str, default=",".join(BANDS))
        return parser

    def __init__(self, args=None):
        super().__init__(args)

        self.n_train_images = self.args.get("n_train_images", N_TRAIN)
        self.n_validation_images = self.args.get("n_validation_images", N_VAL)
        self.bands = self.args.get("--bands", ",".join(BANDS)).split(",")

        if not (os.path.exists(PROCESSED_DATA_FILE_TRAINVAL) and os.path.exists(PROCESSED_DATA_FILE_POOL)):
            _download_and_process_droughtwatch(self)

        self.mapping = list(range(NUM_CLASSES))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, len(self.bands), IMG_DIM, IMG_DIM) # tensor * no. bands * height * width
        self.output_dims = (1,)
        self.prepare_data()
        self.init_setup()

    def prepare_data(self, *args, **kwargs) -> None:
        if not (os.path.exists(PROCESSED_DATA_FILE_TRAINVAL) and os.path.exists(PROCESSED_DATA_FILE_POOL)):
            _download_and_process_droughtwatch(self)

    def init_setup(self):
        print("INIT SETUP DATA CALLED  \n-------------\n")    
        with h5py.File(PROCESSED_DATA_FILE_TRAINVAL, "r") as f:
            self.x_train = f["x_train"][:]
            self.y_train = f["y_train"][:].squeeze().astype(int)
            self.x_val = f["x_val"][:]
            self.y_val = f["y_val"][:].squeeze().astype(int)

        self.data_train = BaseDataset(self.x_train, self.y_train, transform=self.transform)
        self.data_val = BaseDataset(self.x_val, self.y_val, transform=self.transform)
        #self.data_test = self.data_val # NOTE: the framework requires a test set, we just set it to the same as the validation set though
    
        # pool of labeled samples from which to choose from via active learning
        # TODO: do something with this pool
        # NOTE: we might have to change how the pool is stored and read, otherwise Colab's memory is close to its' limit...

        with h5py.File(PROCESSED_DATA_FILE_POOL, "r") as f:
            self.x_pool = f["x_pool"][:]
            self.y_pool = f["y_pool"][:].squeeze().astype(int)

        self.data_test = BaseDataset(self.x_pool, self.y_pool, transform=self.transform) 
        self.data_unlabelled=BaseDataset(self.x_pool, self.y_pool, transform=self.transform)
        print(self)
        print(self.data_unlabelled)
        print(self.data_train)
        print(self.data_val)


    def setup(self, stage: str = None) -> None:

        self.data_train = BaseDataset(self.x_train, self.y_train, transform=self.transform)
        self.data_val = BaseDataset(self.x_val, self.y_val, transform=self.transform)
        self.data_test = BaseDataset(self.x_pool, self.y_pool, transform=self.transform) 
        self.data_unlabelled=BaseDataset(self.x_pool, self.y_pool, transform=self.transform)
        


    def __repr__(self):
        basic = f"DroughtWatch Dataset\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None and self.data_unlabelled is None:
            return basic

        # deepcode ignore unguarded~next~call: <please specify a reason of ignoring this>
        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val sizes: {len(self.data_train)}, {len(self.data_val)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), (x*1.0).mean(), (x*1.0).std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Pool size of labeled samples to do active learning from: {len(self.data_unlabelled)}\n"
        )
        return basic + data

    def get_ds_length(self,ds_name='unlabelled'):
        print(self)
        print(self.data_unlabelled)
        print(self.data_train)
        print(self.data_val)
        if ds_name=='unlabelled':
            return len(self.data_unlabelled.data)
        elif ds_name=='train':
            return len(self.data_train.data)
        elif ds_name=='test' :
            return len(self.data_test.data)
        elif ds_name=='val' :
            return len(self.data_val.data)
        else:
            raise NameError('Unknown Dataset Name '+ds_name) 
           

    def expand_training_set(self, sample_idxs):

        #get x_train, y_train
        x_train=self.data_train.data
        y_train=self.data_train.targets
        #get unlabelled set
        x_pool=self.data_unlabelled.data
        y_pool=self.data_unlabelled.targets

        # get new training examples
        x_train_new = x_pool[sample_idxs]
        y_train_new = y_pool[sample_idxs]

        # remove the new examples from the unlabelled pool
        mask = np.ones(x_pool.shape[0], bool)
        mask[sample_idxs] = False
        self.x_pool = x_pool[mask]
        self.y_pool = y_pool[mask]

        # add new examples to training set
        self.x_train = np.concatenate([x_train, x_train_new])
        self.y_train = np.concatenate([y_train, y_train_new])

        print()

             

def _download_and_process_droughtwatch(self):

    metadata = toml.load(METADATA_FILENAME)

    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(self, metadata["filename"], DL_DATA_DIRNAME)

def _load_data(data_path):

    def file_list_from_folder(folder, data_path):
        folderpath = os.path.join(data_path, folder)
        filelist = []
        for filename in os.listdir(folderpath):
            if filename.startswith('part-') and not filename.endswith('gstmp'):
                filelist.append(os.path.join(folderpath, filename))
        return filelist

    train = file_list_from_folder("train", data_path)
    val = file_list_from_folder("val", data_path)

    return train, val


def _parse_tfrecords(self, filelist, buffer_size, include_viz=False):


    # tf record parsing function
    def _parse_(serialized_example, keylist=self.bands):

        features = {
            'B1': tf.io.FixedLenFeature([], tf.string),
            'B2': tf.io.FixedLenFeature([], tf.string),
            'B3': tf.io.FixedLenFeature([], tf.string),
            'B4': tf.io.FixedLenFeature([], tf.string),
            'B5': tf.io.FixedLenFeature([], tf.string),
            'B6': tf.io.FixedLenFeature([], tf.string),
            'B7': tf.io.FixedLenFeature([], tf.string),
            'B8': tf.io.FixedLenFeature([], tf.string),
            'B9': tf.io.FixedLenFeature([], tf.string),
            'B10': tf.io.FixedLenFeature([], tf.string),
            'B11': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        example = tf.io.parse_single_example(serialized_example, features)
    
        def getband(example_key):
            img = tf.io.decode_raw(example_key, tf.uint8)
            return tf.reshape(img[:IMG_DIM**2], shape=(IMG_DIM, IMG_DIM, 1))
        
        bandlist = [getband(example[key]) for key in keylist]
        
        # combine bands into tensor
        image = tf.concat(bandlist, -1)
        label = tf.cast(example['label'], tf.int32) # NOTE: no one-hot encoding for PyTorch optimizer! (different than in TensorFlow)
    
        # if logging RGB images as examples, generate RGB image from 11-channel satellite image
        if include_viz:
            image = get_img_from_example(example)
            return {'image' : image, 'label': example['label']}, label
        return {'image': image}, label
    
    # create tf dataset from filelist
    tfrecord_dataset = tf.data.TFRecordDataset(filelist)

    # convert the dataset to a dataset of given size
    tfrecord_dataset = tfrecord_dataset.map(lambda x:_parse_(x)).batch(buffer_size) #.shuffle(buffer_size).repeat(-1).batch(buffer_size)
    tfrecord_iterator = iter(tfrecord_dataset)
    #image, label = tfrecord_iterator.get_next()
    #return image, label
    return tfrecord_iterator

def _process_raw_dataset(self, filename: str, dirname: Path):

    print("Unzipping DroughtWatch file...")
    curdir = os.getcwd()
    os.chdir(dirname)

    if not os.path.exists(DL_DATA_DIRNAME / "droughtwatch_data"):
        zip_file = zipfile.ZipFile(filename, "r")
        zip_file.extractall()

    print("Loading train/validation datasets as TF tensor")
    train_tfrecords, val_tfrecords = _load_data(DL_DATA_DIRNAME / "droughtwatch_data")
    train_image_iterator = _parse_tfrecords(self, train_tfrecords, self.n_train_images)
    val_image_iterator = _parse_tfrecords(self, val_tfrecords, self.n_validation_images) 
    train_images, train_labels = train_image_iterator.get_next()
    val_images, val_labels = val_image_iterator.get_next()

    print("Converting train/valildation TF tensors to Numpy")    
    x_train = train_images["image"].numpy()
    y_train = train_labels.numpy()
    x_val = val_images["image"].numpy()
    y_val = val_labels.numpy()

    print("Saving train/validation to HDF5 in compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(PROCESSED_DATA_FILE_TRAINVAL, "w") as f:
        f.create_dataset("x_train", data=x_train, compression="lzf") # take only rows within train_indices
        f.create_dataset("y_train", data=y_train, compression="lzf")
        f.create_dataset("x_val", data=x_val, compression="lzf") # take only n_validation_images rows
        f.create_dataset("y_val", data=y_val, compression="lzf")
    

    print("Saving all remaining labeled images to separate HDF5 pool...")
    pool_images, pool_labels = train_image_iterator.get_next()
    x_pool = pool_images["image"].numpy()
    y_pool = pool_labels.numpy()

    with h5py.File(PROCESSED_DATA_FILE_POOL, "w") as f:
        f.create_dataset("x_pool", data=x_pool, compression="gzip", maxshape=(90000, 65, 65, 11), chunks=True)
        f.create_dataset("y_pool", data=y_pool, compression="gzip", maxshape=(90000,), chunks=True)

    for pool_images, pool_labels in train_image_iterator:
        x_pool = pool_images["image"].numpy()
        y_pool = pool_labels.numpy()

        with h5py.File(PROCESSED_DATA_FILE_POOL, "a") as hf:

            hf["x_pool"].resize((hf["x_pool"].shape[0] + x_pool.shape[0]), axis = 0)
            hf["x_pool"][-x_pool.shape[0]:] = x_pool

            hf["y_pool"].resize((hf["y_pool"].shape[0] + y_pool.shape[0]), axis = 0)
            hf["y_pool"][-y_pool.shape[0]:] = y_pool

    print(PROCESSED_DATA_FILE_POOL)
    print(PROCESSED_DATA_FILE_TRAINVAL,PROCESSED_DATA_DIRNAME)
    print("Cleaning up...")
    shutil.rmtree("droughtwatch_data")
    os.chdir(curdir)

          

if __name__ == "__main__":
    load_and_print_info(DroughtWatch)
