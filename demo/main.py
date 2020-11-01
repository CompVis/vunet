import tensorflow as tf
import os
import math
import pickle
import numpy as np
import PIL
import cv2
import nn
import models
import deeploss

from tqdm import trange
from model import Model
from batches_pg2 import plot_batch
from get_batches import get_batches
from utils import init_logging, process_batches
from parser import parse_arguments
from config import default_log_dir, config, session

if __name__ == "__main__":
    opt = parse_arguments()
    if not os.path.exists(opt.data_index):
        raise Exception("Invalid data index: {}".format(opt.data_index))

    out_dir, logger = init_logging(opt.log_dir)
    logger.info(opt)

    if opt.mode == "train":
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        init_shape = [opt.init_batches * batch_size] + img_shape

        batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = True)
        init_batches = get_batches(init_shape, opt.data_index, mask = opt.mask, train = True)
        valid_batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = False)
        logger.info("Number of training samples: {}".format(batches.n))
        logger.info("Number of validation samples: {}".format(valid_batches.n))
        if valid_batches.n == 0:
            valid_batches = None

        model = Model(opt, out_dir, logger)
        if opt.checkpoint is not None:
            model.restore_graph(opt.checkpoint)
        else:
            model.init_graph(next(init_batches))
        model.fit(batches, valid_batches)

    elif opt.mode == "test":
        if not opt.checkpoint:
            raise Exception("Testing requires --checkpoint")
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        valid_batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = False)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        for i in trange(valid_batches.n // batch_size):
            X_batch, C_batch = next(valid_batches)
            x_gen = model.test(C_batch)
            for k in x_gen:
                plot_batch(x_gen[k], os.path.join(
                    out_dir,
                    "testing_{}_{:07}.png".format(k, i)))
    elif opt.mode == "add_reconstructions":
        if not opt.checkpoint:
            raise Exception("Testing requires --checkpoint")
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        batches = get_batches(data_shape, opt.data_index, mask = opt.mask,
                train = True, return_index_id = True)
        valid_batches = get_batches(data_shape, opt.data_index,
                mask = opt.mask, train = False, return_index_id = True)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        # open index file to get image filenames and update with
        # reconstruction data
        with open(opt.data_index, "rb") as f:
            index = pickle.load(f)
        index_dir = os.path.dirname(opt.data_index)
        index["reconstruction"] = len(index["imgs"]) * [None]
        index["sample"] = len(index["imgs"]) * [None]

        process_batches(model, index_dir, batches, batch_size, index)
        process_batches(model, index_dir, valid_batches, batch_size, index)

        # write updated index
        with open(opt.data_index, "wb") as f:
            pickle.dump(index, f)
        logger.info("Wrote {}".format(opt.data_index))

    elif opt.mode == "transfer":
        if not opt.checkpoint:
            opt.checkpoint = "log/2017-10-25T16:31:50/checkpoints/model.ckpt-100000"
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3] # outputs [256, 256, 3]
        data_shape = [batch_size] + img_shape # outputs [8, 256, 256, 3]
        valid_batches = get_batches(data_shape, opt.data_index,
                mask = opt.mask, train = False)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        ids = ["00038", "00281", "01166", "x", "06909", "y", "07586", "07607", "z", "09874"]
        # for step in trange(10):
        for step in trange(10):
            X_batch, C_batch, XN_batch, CN_batch = next(valid_batches)
            bs = X_batch.shape[0]
            imgs = list()
            imgs.append(np.zeros_like(X_batch[0,...]))
            for r in range(bs):
                imgs.append(C_batch[r,...])
            for i in range(bs):
                x_infer = XN_batch[i,...]
                c_infer = CN_batch[i,...]
                imgs.append(X_batch[i,...])

                x_infer_batch = x_infer[None,...].repeat(bs, axis = 0)
                c_infer_batch = c_infer[None,...].repeat(bs, axis = 0)
                c_generate_batch = C_batch
                results = model.transfer(x_infer_batch, c_infer_batch, c_generate_batch)
                for j in range(bs):
                    imgs.append(results[j,...])
            imgs = np.stack(imgs, axis = 0)
            plot_batch(imgs, os.path.join(
                out_dir,
                "transfer_{}.png".format(ids[step])))

    elif opt.mode == "mcmc":
        if not opt.checkpoint:
            raise Exception("Testing requires --checkpoint")
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        valid_batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = False)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        for i in trange(valid_batches.n // batch_size):
            X_batch, C_batch = next(valid_batches)
            x_gen = model.mcmc(C_batch)
            for k in x_gen:
                plot_batch(x_gen[k], os.path.join(
                    out_dir,
                    "mcmc_{}_{:07}.png".format(k, i)))
    else:
        raise NotImplemented()
