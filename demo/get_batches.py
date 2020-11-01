
import sys
from batches_pg2 import plot_batch
from index_flow import IndexFlow
from buffered_wrapper import BufferedWrapper

def get_batches(
        shape,
        index_path,
        train,
        mask,
        fill_batches = True,
        shuffle = True,
        return_keys = ["imgs", "joints", "norm_imgs", "norm_joints"]):
    """Buffered IndexFlow."""
    flow = IndexFlow(shape, index_path, train, mask, fill_batches, shuffle, return_keys)
    return BufferedWrapper(flow)


if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print("Useage: {} <path to index.p>".format(sys.argv[0]))
        exit(1)

    batches = get_batches(
            shape = (16, 128, 128, 3),
            index_path = sys.argv[1],
            train = True,
            mask = False,
            shuffle = True)
    X, C = next(batches)
    plot_batch(X, "unmasked.png")
    plot_batch(C, "joints.png")

    """
    batches = get_batches(
            shape = (16, 128, 128, 3),
            index_path = sys.argv[1],
            train = True,
            mask = True)
    X, C = next(batches)
    plot_batch(X, "masked.png")

    batches = get_batches(
            shape = (16, 32, 32, 3),
            index_path = sys.argv[1],
            train = True,
            mask = True)
    X, C = next(batches)
    plot_batch(X, "masked32.png")
    plot_batch(C, "joints32.png")
    """
