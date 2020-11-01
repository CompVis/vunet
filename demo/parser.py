import argparse
from config import default_log_dir

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required = True, help = "path to training or testing data index")
    parser.add_argument("--mode", default = "train",
            choices=["train", "test", "mcmc", "add_reconstructions", "transfer"])
    parser.add_argument("--log_dir", default = default_log_dir, help = "path to log into")
    parser.add_argument("--batch_size", default = 8, type = int, help = "batch size")
    parser.add_argument("--init_batches", default = 4, type = int, help = "number of batches for initialization")
    parser.add_argument("--checkpoint", help = "path to checkpoint to restore")
    parser.add_argument("--spatial_size", default = 256, type = int, help = "spatial size to resize images to")
    parser.add_argument("--lr", default = 1e-3, type = float, help = "initial learning rate")
    parser.add_argument("--lr_decay_begin", default = 1000, type = int, help = "steps after which to begin linear lr decay")
    parser.add_argument("--lr_decay_end", default = 100000, type = int, help = "step at which lr is zero, i.e. number of training steps")
    parser.add_argument("--log_freq", default = 250, type = int, help = "frequency to log")
    parser.add_argument("--ckpt_freq", default = 1000, type = int, help = "frequency to checkpoint")
    parser.add_argument("--test_freq", default = 1000, type = int, help = "frequency to test")
    parser.add_argument("--drop_prob", default = 0.1, type = float, help = "Dropout probability")
    parser.add_argument("--mask", dest = "mask", action = "store_true", help = "Use masked data")
    parser.add_argument("--no-mask", dest = "mask", action = "store_false", help = "Do not use mask")
    parser.set_defaults(mask = True)
    return parser.parse_args()

