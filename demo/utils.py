import os
import datetime
import shutil
import glob
import logging
import math
import PIL

from tqdm import trange
from batches_pg2 import postprocess


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


def process_batches(model, index_dir, batches, batch_size, index):
    for i in trange(math.ceil(batches.n / batch_size)):
        X_batch, C_batch, I_batch = next(batches)
        # reconstructions
        R_batch = model.reconstruct(X_batch, C_batch)
        R_batch = postprocess(R_batch) # to uint8 for saving
        # samples from pose
        S_batch = model.test(C_batch)["test_sample"]
        S_batch = postprocess(S_batch) # to uint8 for saving
        for batch_i, i in enumerate(I_batch):
            original_fname = index["imgs"][i]
            reconstr_fname = original_fname.rsplit(".", 1)[0] + "_reconstruction.png"
            reconstr_path = os.path.join(index_dir, reconstr_fname)
            sample_fname = original_fname.rsplit(".", 1)[0] + "_sample.png"
            sample_path = os.path.join(index_dir, sample_fname)
            index["reconstruction"][i] = reconstr_path
            index["sample"][i] = sample_path
            PIL.Image.fromarray(R_batch[batch_i,...]).save(reconstr_path)
            PIL.Image.fromarray(S_batch[batch_i,...]).save(sample_path)

