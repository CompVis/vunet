import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)

import os, logging, shutil, datetime, time, math
import glob
import argparse
import numpy as np
from tqdm import tqdm, trange

import nn
import models
from batches import get_batches, plot_batch


def init_logging(out_base_dir):
    # get unique output directory based on current time
    os.makedirs(out_base_dir, exist_ok = True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = os.path.join(out_base_dir, now)
    os.makedirs(out_dir, exist_ok = False)
    # copy source code to logging dir to have an idea what the run was about
    this_file = os.path.realpath(__file__)
    assert(this_file.endswith(".py"))
    shutil.copy(this_file, out_dir)
    # copy all py files to logging dir
    src_dir = os.path.dirname(this_file)
    py_files = glob.glob(os.path.join(src_dir, "*.py"))
    for py_file in py_files:
        shutil.copy(py_file, out_dir)
    # init logging
    logging.basicConfig(filename = os.path.join(out_dir, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return out_dir, logger


class Timer(object):
    def __init__(self):
        self.tick()


    def tick(self):
        self.start_time = time.time()


    def tock(self):
        self.end_time = time.time()
        time_since_tick = self.end_time - self.start_time
        self.tick()
        return time_since_tick


class Model(object):
    def __init__(self, opt, out_dir, logger):
        self.batch_size = opt.batch_size
        self.img_shape = 2*[opt.spatial_size] + [3]
        self.init_batches = opt.init_batches

        self.initial_lr = opt.lr
        self.lr_decay_begin = opt.lr_decay_begin
        self.lr_decay_end = opt.lr_decay_end

        self.out_dir = out_dir
        self.logger = logger
        self.log_frequency = opt.log_freq
        self.ckpt_frequency = opt.ckpt_freq
        self.test_frequency = opt.test_freq
        self.checkpoint_best = False

        self.dropout_p = opt.drop_prob
        self.n_scales = opt.n_scales

        self.best_loss = float("inf")
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        self.define_models()
        self.define_graph()


    def define_models(self):
        self.enc_up_pass = models.make_model(
                "enc_up", models.enc_up,
                n_scales = self.n_scales)
        self.enc_down_pass = models.make_model(
                "enc_down", models.enc_down,
                n_scales = self.n_scales)
        self.dec_up_pass = models.make_model(
                "dec_up", models.dec_up,
                n_scales = self.n_scales)
        self.dec_down_pass = models.make_model(
                "dec_down", models.dec_down,
                n_scales = self.n_scales)
        self.dec_params = models.make_model(
                "dec_params", models.dec_parameters)


    def train_forward_pass(self, x, c, dropout_p, init = False):
        kwargs = {"init": init, "dropout_p": dropout_p}
        # encoder
        hs = self.enc_up_pass(x, c, **kwargs)
        es, qs, zs_posterior = self.enc_down_pass(hs, **kwargs)
        # decoder
        gs = self.dec_up_pass(c, **kwargs)
        ds, ps, zs_prior = self.dec_down_pass(gs, zs_posterior, training = True, **kwargs)
        params = self.dec_params(ds[-1], **kwargs)
        activations = hs + es + gs + ds
        return params, qs, ps, activations


    def test_forward_pass(self, c):
        kwargs = {"init": False, "dropout_p": 0.0}
        # decoder
        gs = self.dec_up_pass(c, **kwargs)
        ds, ps, zs_prior = self.dec_down_pass(gs, [], training = False, **kwargs)
        params = self.dec_params(ds[-1], **kwargs)
        return params


    def sample(self, params):
        return models.dec_sample(params)


    def likelihood_loss(self, x, params):
        return nn.likelihood_loss(x, params, "h1")


    def define_graph(self):
        global_step = tf.Variable(0, trainable = False, name = "global_step")
        lr = nn.make_linear_var(
                global_step,
                self.lr_decay_begin, self.lr_decay_end,
                self.initial_lr, 0.0,
                0.0, self.initial_lr)
        kl_weight = nn.make_linear_var(
                global_step,
                self.lr_decay_begin, self.lr_decay_end // 2,
                0.0, 1.0,
                1e-5, 1.0)

        # initialization
        self.x_init = tf.placeholder(
                tf.float32,
                shape = [self.init_batches * self.batch_size] + self.img_shape)
        self.c_init = tf.placeholder(
                tf.float32,
                shape = [self.init_batches * self.batch_size] + self.img_shape)
        _ = self.train_forward_pass(self.x_init, self.c_init, dropout_p = self.dropout_p, init = True)

        # training
        self.x = tf.placeholder(
                tf.float32,
                shape = [self.batch_size] + self.img_shape)
        self.c = tf.placeholder(
                tf.float32,
                shape = [self.batch_size] + self.img_shape)
        # compute parameters of model distribution
        params, qs, ps, activations = self.train_forward_pass(self.x, self.c, dropout_p = self.dropout_p)
        # sample from model distribution
        sample = self.sample(params)
        # maximize likelihood
        loss = self.likelihood_loss(self.x, params)
        for q, p in zip(qs, ps):
            loss += kl_weight * models.latent_kl(q, p)

        # testing
        test_sample = self.sample(self.test_forward_pass(self.c))

        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5, beta2 = 0.9)
        opt_op = optimizer.minimize(loss, var_list = tf.trainable_variables())
        self.train_op = opt_op
        with tf.control_dependencies([self.train_op]):
            self.train_op = tf.assign(global_step, global_step + 1)

        # logging and visualization
        self.log_ops = dict()
        self.log_ops["global_step"] = global_step
        self.log_ops["kl_weight"] = kl_weight
        self.log_ops["loss"] = loss
        self.img_ops = dict()
        self.img_ops["sample"] = sample
        self.img_ops["test_sample"] = test_sample
        self.img_ops["x"] = self.x
        self.img_ops["c"] = self.c

        # keep seperate train and validation summaries
        # only training summary contains histograms
        train_summaries = list()
        for k, v in self.log_ops.items():
            train_summaries.append(tf.summary.scalar(k, v))
        self.train_summary_op = tf.summary.merge_all()

        valid_summaries = list()
        for k, v in self.log_ops.items():
            valid_summaries.append(tf.summary.scalar(k+"_valid", v))
        self.valid_summary_op = tf.summary.merge(valid_summaries)

        self.logger.info("Defined graph")


    def init_graph(self, init_batch):
        self.writer = tf.summary.FileWriter(
                self.out_dir,
                session.graph)
        self.saver = tf.train.Saver()
        session.run(tf.global_variables_initializer(), {
            self.x_init: init_batch[0],
            self.c_init: init_batch[1]})
        self.logger.info("Initialized model from scratch")


    def restore_graph(self, restore_path):
        self.writer = tf.summary.FileWriter(
                self.out_dir,
                session.graph)
        self.saver = tf.train.Saver()
        self.saver.restore(session, restore_path)
        self.logger.info("Restored model from {}".format(restore_path))


    def fit(self, batches, valid_batches = None):
        start_step = self.log_ops["global_step"].eval(session)
        self.valid_batches = valid_batches
        for batch in trange(start_step, self.lr_decay_end):
            X_batch, C_batch = next(batches)
            feed_dict = {
                    self.x: X_batch,
                    self.c: C_batch}
            fetch_dict = {"train": self.train_op}
            if self.log_ops["global_step"].eval(session) % self.log_frequency == 0:
                fetch_dict["log"] = self.log_ops
                fetch_dict["img"] = self.img_ops
                fetch_dict["summary"] = self.train_summary_op
            result = session.run(fetch_dict, feed_dict)
            self.log_result(result)


    def log_result(self, result, **kwargs):
        global_step = self.log_ops["global_step"].eval(session)
        if "summary" in result:
            self.writer.add_summary(result["summary"], global_step)
            self.writer.flush()
        if "log" in result:
            for k, v in result["log"].items():
                self.logger.info("{}: {}".format(k, v))
        if "img" in result:
            for k, v in result["img"].items():
                plot_batch(v, os.path.join(
                    self.out_dir,
                    k + "_{:07}.png".format(global_step)))

            if self.valid_batches is not None:
                # validation run
                X_batch, C_batch = next(self.valid_batches)
                feed_dict = {
                        self.x: X_batch,
                        self.c: C_batch}
                fetch_dict = dict()
                fetch_dict["imgs"] = self.img_ops
                fetch_dict["summary"] = self.valid_summary_op
                fetch_dict["validation_loss"] = self.log_ops["loss"]
                result = session.run(fetch_dict, feed_dict)
                self.writer.add_summary(result["summary"], global_step)
                self.writer.flush()
                # display samples
                imgs = result["imgs"]
                for k, v in imgs.items():
                    plot_batch(v, os.path.join(
                        self.out_dir,
                        "valid_" + k + "_{:07}.png".format(global_step)))
                # log validation loss
                validation_loss = result["validation_loss"]
                self.logger.info("{}: {}".format("validation_loss", validation_loss))
                if self.checkpoint_best and validation_loss < self.best_loss:
                    # checkpoint if validation loss improved
                    self.logger.info("step {}: Validation loss improved from {:.4e} to {:.4e}".format(global_step, self.best_loss, validation_loss))
                    self.best_loss = validation_loss
                    self.make_checkpoint(global_step, prefix = "best_")
        if global_step % self.test_frequency == 0:
            # testing
            if self.valid_batches is not None:
                X_batch, C_batch = next(self.valid_batches)
                x_gen = self.test(C_batch)
                for k in x_gen:
                    plot_batch(x_gen[k], os.path.join(
                        self.out_dir,
                        "testing_{}_{:07}.png".format(k, global_step)))
        if global_step % self.ckpt_frequency == 0:
            self.make_checkpoint(global_step)


    def make_checkpoint(self, global_step, prefix = ""):
        fname = os.path.join(self.checkpoint_dir, prefix + "model.ckpt")
        self.saver.save(
                session,
                fname,
                global_step = global_step)
        self.logger.info("Saved model to {}".format(fname))


    def test(self, c_batch):
        results = dict()
        results["cond"] = c_batch
        results["test_sample"] = session.run(self.img_ops["test_sample"], {self.c: c_batch})
        return results


if __name__ == "__main__":
    default_log_dir = os.path.join(os.getcwd(), "log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required = True, help = "path to training or testing data index")
    parser.add_argument("--mode", default = "train", choices=["train", "test"])
    parser.add_argument("--log_dir", default = default_log_dir, help = "path to log into")
    parser.add_argument("--batch_size", default = 16, type = int, help = "batch size")
    parser.add_argument("--init_batches", default = 8, type = int, help = "number of batches for initialization")
    parser.add_argument("--checkpoint", help = "path to checkpoint to restore")
    parser.add_argument("--spatial_size", default = 32, type = int, help = "spatial size to resize images to")
    parser.add_argument("--lr", default = 1e-3, type = float, help = "initial learning rate")
    parser.add_argument("--lr_decay_begin", default = 1000, type = int, help = "steps after which to begin linear lr decay")
    parser.add_argument("--lr_decay_end", default = 100000, type = int, help = "step at which lr is zero, i.e. number of training steps")
    parser.add_argument("--log_freq", default = 250, type = int, help = "frequency to log")
    parser.add_argument("--ckpt_freq", default = 1000, type = int, help = "frequency to checkpoint")
    parser.add_argument("--test_freq", default = 1000, type = int, help = "frequency to test")
    parser.add_argument("--n_scales", default = 3, type = int, help = "Number of scales")
    parser.add_argument("--drop_prob", default = 0.5, type = float, help = "Dropout probability")
    parser.add_argument("--mask", dest = "mask", action = "store_true", help = "Use masked data")
    parser.add_argument("--no-mask", dest = "mask", action = "store_false", help = "Do not use mask")
    parser.set_defaults(mask = True)
    opt = parser.parse_args()

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
    else:
        raise NotImplemented()
