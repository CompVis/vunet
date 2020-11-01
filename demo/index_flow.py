import numpy as np
import pickle
import os

from batches_pg2 import (
    valid_joints,
    load_img,
    preprocess,
    preprocess_mask,
    make_joint_img,
    normalize,
    make_mask_img
)

class IndexFlow(object):
    """Batches from index file."""

    def __init__(
            self,
            shape,
            index_path,
            train,
            mask = True,
            fill_batches = True,
            shuffle = True,
            return_keys = ["imgs", "joints"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        self.basepath = os.path.dirname(index_path)
        self.train = train
        self.mask = mask
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle
        self.return_keys = return_keys

        self.jo = self.index["joint_order"]
        # rescale joint coordinates to image shape
        h,w = self.img_shape[:2]
        wh = np.array([[[w,h]]])
        self.index["joints"] = self.index["joints"] * wh

        self.indices = np.array(
                [i for i in range(len(self.index["train"]))
                    if self._filter(i)])

        self.n = self.indices.shape[0]
        self.shuffle()


    def _filter(self, i):
        good = True
        good = good and (self.index["train"][i] == self.train)
        joints = self.index["joints"][i]
        required_joints = ["lshoulder","rshoulder","lhip","rhip"]
        joint_indices = [self.jo.index(b) for b in required_joints]
        joints = np.float32(joints[joint_indices])
        good = good and valid_joints(joints)
        return good


    def __next__(self):
        batch = dict()

        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis = 0)
            assert(batch_indices.shape[0] == self.batch_size)
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        # prepare batch data
        # load images
        batch["imgs"] = list()
        for i in batch_indices:
            fname = self.index["imgs"][i]
            # traintest = "train" if self.train else "test"
            # path = os.path.join(self.basepath, "..", "original", "filted_up_{}".format(traintest), fname)
            path = os.path.join(self.basepath, fname)
            batch["imgs"].append(load_img(path, target_size = self.img_shape))
        batch["imgs"] = np.stack(batch["imgs"])
        batch["imgs"] = preprocess(batch["imgs"])

        # load joint coordinates
        batch["joints_coordinates"] = [self.index["joints"][i] for i in batch_indices]

        # generate stickmen images from coordinates
        batch["joints"] = list()
        for joints in batch["joints_coordinates"]:
            img = make_joint_img(self.img_shape, self.jo, joints)
            batch["joints"].append(img)
        batch["joints"] = np.stack(batch["joints"])
        batch["joints"] = preprocess(batch["joints"])

        if False and self.mask:
            if "masks" in self.index:
                batch_masks = list()
                for i in batch_indices:
                    fname = self.index["masks"][i]
                    path = os.path.join(self.basepath, fname)
                    batch_masks.append(load_img(path, target_size = self.img_shape))
            else:
                # generate mask based on joint coordinates
                batch_masks = list()
                for joints in batch["joints_coordinates"]:
                    mask = make_mask_img(self.img_shape, self.jo, joints)
                    batch_masks.append(mask)
            batch["masks"] = np.stack(batch_masks)
            batch["masks"] = preprocess_mask(batch["masks"])
            # apply mask to images
            batch["imgs"] = batch["imgs"] * batch["masks"]


        imgs, joints = normalize(batch["imgs"], batch["joints_coordinates"], batch["joints"], self.jo)
        batch["norm_imgs"] = imgs
        batch["norm_joints"] = joints

        batch_list = [batch[k] for k in self.return_keys]
        return batch_list


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


