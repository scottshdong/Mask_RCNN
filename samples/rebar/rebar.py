"""
Mask R-CNN
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 rebar.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 rebar.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 rebar.py train --dataset=/path/to/balloon/dataset --weights=imagenet
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")

# Root directory of the project
# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# Go up two levels to the repo root
ROOT_DIR = os.path.join(ROOT_DIR, "..", "..")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/rebar/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
    "newset4-2_00000",
    "newset4-2_00001",
    "newset4-2_00002",
    "newset4-2_00003",
    "newset4-2_00004",
    "newset4-2_00005",
    "newset4-2_00006",
    "newset4-2_00007",
    "newset4-2_00008",
    "newset4-2_00009",
    "newset4-2_00010",
    "newset4-2_00011",
    "newset4-2_00012",
    "newset4-2_00013",
    "newset4-2_00014",
    "newset4-2_00015",
    "newset1_00001",
    "newset1_00002",
    "newset1_00003",
    "newset1_00004",
    "newset1_00005",
    "newset1_00006",
    "newset1_00007",
    "newset1_00008",
    "newset1_00009",
    "newset1_00010",
    "newset1_00011",
    "newset1_00012",
    "newset1_00013",
    "newset1_00014",
    "newset1_00015",
    "bar_00001",
    "bar_00002",
    "bar_00003",
    "bar_00004",
    "bar_00005",
    "bar_00006",
    "bar_00007",
    "bar_00008",
    "bar_00009",
    "bar_00010",
    "bar_00011",
    "bar_00012",
    "bar_00013",
    "bar_00014",
    "bar_00015",
]

TRAIN_PORT = 0.8

############################################################
#  Configurations
############################################################

class RebarConfig(Config):
    """Configuration for training on the rebar segmentation dataset."""
    # Give the configuration a recognizable name
    # Give the configuration a recognizable name
    NAME = "rebar"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0
    DETECTION_MAX_INSTANCES = 50


class RebarInferenceConfig(RebarConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6


############################################################
#  Dataset
############################################################

class RebarDataset(utils.Dataset):

    def load_rebar(self, dataset_dir, subset):
        """Load a subset of the rebar dataset.

        dataset_dir: Root directory of the dataset
        subset: * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset rebar, and the class rebar
        self.add_class("rebar", 1, "rebar")
        img_subdir = 'img'
        image_dir = os.path.join(dataset_dir, img_subdir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            # image_ids = os.walk(image_dir)[2]
            image_ids = []
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.startswith('bar') \
                        or file.startswith('newset'):
                            name, ext = os.path.splitext(file)
                            image_ids.append(name)

            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))
                # mixed bar and newset
                newset_image_ids = [
                    image_id
                    for image_id in image_ids if image_id.startswith('newset')
                ]

                bar_image_ids = [
                    image_id
                    for image_id in image_ids if image_id.startswith('bar')
                ]

                np.random.shuffle(newset_image_ids)
                np.random.shuffle(bar_image_ids)

                newset_len = len(newset_image_ids)
                bar_len = len(bar_image_ids)

                image_ids = []
                image_ids.extend(bar_image_ids[0:int(bar_len*TRAIN_PORT)])
                image_ids.extend(newset_image_ids[0:int(newset_len*TRAIN_PORT)])

        # detect
        train_image_ids = set()
        valid_image_ids = set(image_ids)
        mask_output_dir = os.path.join(image_dir, '..', "mask_output")
        for root, dirs, files in os.walk(mask_output_dir):
            for file in files:
                name, ext = os.path.splitext(file)
                id = '_'.join(name.split('_')[:-1])
                if id in valid_image_ids and id not in train_image_ids:
                    train_image_ids.add(id)

        # Add images
        for image_id in train_image_ids:
            self.add_image(
                "rebar",
                image_id=image_id,
                path=os.path.join(image_dir, "{}.{}".format(
                    image_id, 'png' if image_id.startswith('bar') else 'jpg'
                ))
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "rebar":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(info['path']), '..' , "mask_output")

        # Read mask files from png image
        image_name = info['id']
        mask = []
        for root, dirs, files in os.walk(mask_dir):
            for f in files:
                if f.startswith(image_name):
                    m = skimage.io.imread(os.path.join(mask_dir, f)).astype(bool)
                    mask.append(m)

        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "rebar":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = RebarDataset()
    dataset_train.load_rebar(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RebarDataset()
    dataset_val.load_rebar(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    # augmentation = iaa.SomeOf((0, 2), [
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.OneOf([iaa.Affine(rotate=90),
    #                iaa.Affine(rotate=180),
    #                iaa.Affine(rotate=270)]),
    #     iaa.Multiply((0.8, 1.5)),
    #     iaa.GaussianBlur(sigma=(0.0, 5.0))
    # ])

    # *** This training schedule is an example. Update to your needs ***
    augmentation = iaa.Fliplr(0.5)

    # Training - Stage 1
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                augmentation=augmentation,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=35,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=60,
                layers='all',
                augmentation=augmentation)

############################################################
#  Detection
############################################################
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None):
    assert image_path

    # Image
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = RebarConfig()
    else:
        config = RebarInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, 'train')
    elif args.command == "detect":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
