# A Variational U-Net for Conditional Appearance and Shape Generation

This repository contains training code for the CVPR 2018 spotlight

[**A Variational U-Net for Conditional Appearance and Shape Generation**](https://compvis.github.io/vunet/images/vunet.pdf)

The model learns to infer appearance from a single image and can synthesize
images with that appearance in different poses.

![teaser](assets/cvpr2018_large.gif)

[Project page with more results](https://compvis.github.io/vunet/)

## Requirements

The code was developed with Python 3. Dependencies can be installed with

    pip install -r requirements.txt

Please note that the code does not work with `tensorflow >= 1.3.0`. 

## Training

[Download](http://129.206.117.181:8080/) and unpack the desired dataset.
This results in a folder containing an `index.p` file. Either add a symbolic
link named `data` pointing to the download directory or adjust the path to
the `index.p` file in the `<dataset>.yaml` config file.

For convenience, you can also run

    ./download_data.sh <dataset> <store_dir>

which will perform the above steps automatically. `<dataset>` can be one of
`coco`, `deepfashion` or `market`. To train the model, run

    python main.py --config <dataset>.yaml

By default, images and checkpoints are saved to `log/<current date>`. To
change the log directory and other options, see

    python main.py -h

and the corresponding configuration file. To obtain images of optimal
quality it is recommended to train for a second round with a loss based on
Gram matrices. To do so run

    python main.py --config <dataset>_retrain.yaml --retrain --checkpoint <path to checkpoint of first round>


## Other Datasets

To be able to train the model on your own dataset you must provide a pickled
dictionary with the following keys:

- `joint_order`: list indicating the order of joints. 
- `imgs`: list of paths to images (relative to pickle file).
- `train`: list of booleans indicating if this image belongs to training split
- `joints`: list of `[0,1]` normalized xy joint coordinates of shape `(len(joint_jorder), 2)`. Use negative values for occluded joints.

`joint_order` should contain

    'rankle', 'rknee', 'rhip', 'rshoulder', 'relbow', 'rwrist', 'reye', 'lankle', 'lknee', 'lhip', 'lshoulder', 'lelbow', 'lwrist', 'leye', 'cnose'

and images without valid values for `rhip, rshoulder, lhip, lshoulder` are
ignored.
