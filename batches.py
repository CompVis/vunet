import PIL.Image
from multiprocessing.pool import ThreadPool
import numpy as np
import pickle
import os
import cv2
import math


class BufferedWrapper(object):
    """Fetch next batch asynchronuously to avoid bottleneck during GPU
    training."""
    def __init__(self, gen):
        self.gen = gen
        self.n = gen.n
        self.pool = ThreadPool(1)
        self._async_next()


    def _async_next(self):
        self.buffer_ = self.pool.apply_async(next, (self.gen,))


    def __next__(self):
        result = self.buffer_.get()
        self._async_next()
        return result


def load_img(path, target_size):
    """Load image. target_size is specified as (height, width, channels)
    where channels == 1 means grayscale. uint8 image returned."""
    img = PIL.Image.open(path)
    grayscale = target_size[2] == 1
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    wh_tuple = (target_size[1], target_size[0])
    if img.size != wh_tuple:
        img = img.resize(wh_tuple, resample = PIL.Image.BILINEAR)

    x = np.asarray(img, dtype = "uint8")
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)

    return x


def save_image(X, name):
    """Save image as png."""
    fname = os.path.join(out_dir, name + ".png")
    PIL.Image.fromarray(X).save(fname)


def preprocess(x):
    """From uint8 image to [-1,1]."""
    return np.cast[np.float32](x / 127.5 - 1.0)


def preprocess_mask(x):
    """From uint8 mask to [0,1]."""
    mask = np.cast[np.float32](x / 255.0)
    if mask.shape[-1] == 3:
        mask = np.amax(mask, axis = -1, keepdims = True)
    return mask


def postprocess(x):
    """[-1,1] to uint8."""
    x = (x + 1.0) / 2.0
    x = np.clip(255 * x, 0, 255)
    x = np.cast[np.uint8](x)
    return x


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def plot_batch(X, out_path):
    """Save batch of images tiled."""
    X = postprocess(X)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    PIL.Image.fromarray(canvas).save(out_path)


def make_joint_img(img_shape, jo, joints):
    # three channels: left, right, center
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype = "uint8"))

    if "chead" in jo:
        # MPII
        thickness = 3

        # fill body
        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part),:] for part in body]])
        body_pts = np.int_(body_pts)
        cv2.fillPoly(imgs[2], body_pts, 255)

        right_lines = [
                ("rankle", "rknee"),
                ("rknee", "rhip"),
                ("rhip", "rshoulder"),
                ("rshoulder", "relbow"),
                ("relbow", "rwrist")]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            a = tuple(np.int_(joints[l[0]]))
            b = tuple(np.int_(joints[l[1]]))
            cv2.line(imgs[0], a, b, color = 255, thickness = thickness)

        left_lines = [
                ("lankle", "lknee"),
                ("lknee", "lhip"),
                ("lhip", "lshoulder"),
                ("lshoulder", "lelbow"),
                ("lelbow", "lwrist")]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            a = tuple(np.int_(joints[l[0]]))
            b = tuple(np.int_(joints[l[1]]))
            cv2.line(imgs[1], a, b, color = 255, thickness = thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("chead")]
        neck = 0.5*(rs+ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        cv2.line(imgs[0], a, b, color = 127, thickness = thickness)
        cv2.line(imgs[1], a, b, color = 127, thickness = thickness)
    else:
        assert("cnose" in jo)
        # MSCOCO
        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part),:] for part in body]])
        if np.min(body_pts) >= 0:
            body_pts = np.int_(body_pts)
            cv2.fillPoly(imgs[2], body_pts, 255)

        thickness = 3
        right_lines = [
                ("rankle", "rknee"),
                ("rknee", "rhip"),
                ("rhip", "rshoulder"),
                ("rshoulder", "relbow"),
                ("relbow", "rwrist")]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[0], a, b, color = 255, thickness = thickness)

        left_lines = [
                ("lankle", "lknee"),
                ("lknee", "lhip"),
                ("lhip", "lshoulder"),
                ("lshoulder", "lelbow"),
                ("lelbow", "lwrist")]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[1], a, b, color = 255, thickness = thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("cnose")]
        neck = 0.5*(rs+ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        cv2.line(imgs[0], a, b, color = 127, thickness = thickness)
        cv2.line(imgs[1], a, b, color = 127, thickness = thickness)

    img = np.stack(imgs, axis = -1)
    if img_shape[-1] == 1:
        img = np.mean(img, axis = -1)[:,:,None]
    return img


def make_mask_img(img_shape, jo, joints):
    masks = 3*[None]
    for i in range(3):
        masks[i] = np.zeros(img_shape[:2], dtype = "uint8")

    body = ["lhip", "lshoulder", "rshoulder", "rhip"]
    body_pts = np.array([[joints[jo.index(part),:] for part in body]], dtype = np.int32)
    cv2.fillPoly(masks[1], body_pts, 255)

    head = ["lshoulder", "chead", "rshoulder"]
    head_pts = np.array([[joints[jo.index(part),:] for part in head]], dtype = np.int32)
    cv2.fillPoly(masks[2], head_pts, 255)

    thickness = 15
    lines = [[
        ("rankle", "rknee"),
        ("rknee", "rhip"),
        ("rhip", "lhip"),
        ("lhip", "lknee"),
        ("lknee", "lankle") ], [
            ("rhip", "rshoulder"),
            ("rshoulder", "relbow"),
            ("relbow", "rwrist"),
            ("rhip", "lhip"),
            ("rshoulder", "lshoulder"),
            ("lhip", "lshoulder"),
            ("lshoulder", "lelbow"),
            ("lelbow", "lwrist")], [
                ("rshoulder", "chead"),
                ("rshoulder", "lshoulder"),
                ("lshoulder", "chead")]]
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            line = [jo.index(lines[i][j][0]), jo.index(lines[i][j][1])]
            a = tuple(np.int_(joints[line[0]]))
            b = tuple(np.int_(joints[line[1]]))
            cv2.line(masks[i], a, b, color = 255, thickness = thickness)

    for i in range(3):
        r = 11
        masks[i] = cv2.GaussianBlur(masks[i], (r,r), 0)
        maxmask = np.max(masks[i])
        if maxmask > 0:
            masks[i] = masks[i] / maxmask
    mask = np.stack(masks, axis = -1)
    mask = np.uint8(255 * mask)

    return mask


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
            return_index_id = False):
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
        self.return_index_id = return_index_id

        self.jo = self.index["joint_order"]
        self.indices = np.array(
                [i for i in range(len(self.index["train"]))
                    if self.index["train"][i] == self.train])
        # rescale joint coordinates to image shape
        self.index["joints"] = [v * self.img_shape[0] / 256 for v in self.index["joints"]]

        self.n = self.indices.shape[0]
        self.shuffle()


    def __next__(self):
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis = 0)
            assert(batch_indices.shape[0] == self.batch_size)
        batch_indices = np.array(batch_indices)

        keys = ["imgs", "joints"]
        batch = dict()
        for k in keys:
            batch[k] = [self.index[k][i] for i in batch_indices]

        # load images
        batch_imgs = list()
        for fname in batch["imgs"]:
            path = os.path.join(self.basepath, fname)
            batch_imgs.append(load_img(path, target_size = self.img_shape))
        batch["imgs"] = np.stack(batch_imgs)

        # generate mask based on joint locations
        if self.mask:
            batch_masks = list()
            for joints in batch["joints"]:
                mask = make_mask_img(self.img_shape, self.jo, joints)
                batch_masks.append(mask)
            batch["masks"] = np.stack(batch_masks)

        # generate stickmen
        batch_joints = list()
        for joints in batch["joints"]:
            img = make_joint_img(self.img_shape, self.jo, joints)
            batch_joints.append(img)
        batch["joints"] = np.stack(batch_joints)

        # prepare new epoch
        if batch_end >= self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        # preprocessing
        batch["imgs"] = preprocess(batch["imgs"])
        if self.mask:
            batch["masks"] = preprocess_mask(batch["masks"])
            batch["imgs"] = batch["imgs"] * batch["masks"]
        batch["joints"] = preprocess(batch["joints"])

        if self.return_index_id:
            return batch["imgs"], batch["joints"], batch_indices
        else:
            return batch["imgs"], batch["joints"]


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


def get_batches(
        shape,
        index_path,
        train,
        mask,
        fill_batches = True,
        shuffle = True,
        return_index_id = False):
    """Buffered IndexFlow."""
    flow = IndexFlow(shape, index_path, train, mask, fill_batches, shuffle, return_index_id)
    return BufferedWrapper(flow)


if __name__ == "__main__":
    import sys
    if not len(sys.argv) == 2:
        print("Useage: {} <path to index.p>".format(sys.argv[0]))
        exit(1)

    batches = get_batches(
            shape = (16, 128, 128, 3),
            index_path = sys.argv[1],
            train = True,
            mask = False)
    X, C = next(batches)
    plot_batch(X, "unmasked.png")
    plot_batch(C, "joints.png")

    batches = get_batches(
            shape = (16, 128, 128, 3),
            index_path = sys.argv[1],
            train = True,
            mask = True)
    X, C = next(batches)
    plot_batch(X, "masked.png")
