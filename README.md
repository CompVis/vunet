# A Variational U-Net for Conditional Appearance and Shape Generation

This repository contains code for the paper

**A Variational U-Net for Conditional Appearance and Shape Generation**

The model learns to infer appearance from a single image and can synthesize
images with that appearance in different poses.

![teaser](assets/cvpr2018_large.gif)

[HQ](https://gfycat.com/ThinUntriedGoldenmantledgroundsquirrel)

## Requirements

The code was developed with Python 3. Dependencies can be installed with

    pip install -r requirements.txt # TODO

Please note that the code does not work with `tensorflow >= 1.3.0`. 

## Training

[Download](TODO) and unpack the desired dataset. This results in a folder
containing a `index.p` file. To train the model, run

    python main.py --data_index <path_to_index.p>

By default this saves images and checkpoints to `log/<current date>`. To
adapt log directories and other options, see

    python main.py -h

To obtain images of optimal quality it is recommended to train for a second
round with a loss based on Gram matrices. To do so run

    python main.py --data_index <path_to_index.p> --checkpoint <path to checkpoint of first round> --retrain --gram


## Other Datasets

To be able to train the model on your own dataset you must provide a pickled
dictionary with the following keys:

- `joint_order`: list indicating the order of joints. 
- `imgs`: list of paths to images (relative to pickle file).
- `train`: list of booleans indicating if this image belongs to training split
- `joints`: list of `[0,1]` normalized xy joint coordinates of shape `(len(joint_jorder), 2)`. Use negative values for occluded joints.

`joint_order` should contain

    'ranke', 'rknee', 'rhip', 'rshoulder', 'relbow', 'rwrist', 'reye', 'lanke', 'lknee', 'lhip', 'lshoulder', 'lelbow', 'lwrist', 'leye', 'cnose'
