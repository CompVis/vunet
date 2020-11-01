from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2
import math

def load_img(path, target_size):
    """Load image. target_size is specified as (height, width, channels)
    where channels == 1 means grayscale. uint8 image returned."""
    img = Image.open(path)
    grayscale = target_size[2] == 1
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    wh_tuple = (target_size[1], target_size[0])
    if img.size != wh_tuple:
        img = img.resize(wh_tuple, resample = Image.BILINEAR)

    x = np.asarray(img, dtype = "uint8")
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)

    return x


def save_image(X, name):
    """Save image as png."""
    fname = os.path.join(out_dir, name + ".png")
    Image.fromarray(X).save(fname)


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
    n_channels = X.shape[3]
    if n_channels > 3:
        X = X[:,:,:,np.random.choice(n_channels, size = 3)]
    X = postprocess(X)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    Image.fromarray(canvas).save(out_path)


def make_joint_img(img_shape, jo, joints):
    # three channels: left, right, center
    scale_factor = img_shape[1] / 128
    thickness = int(3 * scale_factor)
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype = "uint8"))

    assert("cnose" in jo)
    # MSCOCO
    body = ["lhip", "lshoulder", "rshoulder", "rhip"]
    body_pts = np.array([[joints[jo.index(part),:] for part in body]])
    if np.min(body_pts) >= 0:
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
    if np.min(a) >= 0 and np.min(b) >= 0:
        cv2.line(imgs[0], a, b, color = 127, thickness = thickness)
        cv2.line(imgs[1], a, b, color = 127, thickness = thickness)

    cn = tuple(np.int_(cn))
    leye = tuple(np.int_(joints[jo.index("leye")]))
    reye = tuple(np.int_(joints[jo.index("reye")]))
    if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
        cv2.line(imgs[0], cn, reye, color = 255, thickness = thickness)
        cv2.line(imgs[1], cn, leye, color = 255, thickness = thickness)

    img = np.stack(imgs, axis = -1)
    if img_shape[-1] == 1:
        img = np.mean(img, axis = -1)[:,:,None]
    return img


def valid_joints(*joints):
    j = np.stack(joints)
    return (j >= 0).all()


def zoom(img, factor, center = None):
    shape = img.shape[:2]
    if center is None or not valid_joints(center):
        center = np.array(shape) / 2
    e1 = np.array([1,0])
    e2 = np.array([0,1])

    dst_center = np.array(center)
    dst_e1 = e1 * factor
    dst_e2 = e2 * factor

    src = np.float32([center, center+e1, center+e2])
    dst = np.float32([dst_center, dst_center+dst_e1, dst_center+dst_e2])
    M = cv2.getAffineTransform(src, dst)

    return cv2.warpAffine(img, M, shape, flags = cv2.INTER_AREA, borderMode = cv2.BORDER_REPLICATE)


def get_crop(bpart, joints, jo, wh, o_w, o_h, ar = 1.0):
    bpart_indices = [jo.index(b) for b in bpart]
    part_src = np.float32(joints[bpart_indices])

    # fall backs
    if not valid_joints(part_src):
        if bpart[0] == "lhip" and bpart[1] == "lknee":
            bpart = ["lhip"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])
        elif bpart[0] == "rhip" and bpart[1] == "rknee": 
            bpart = ["rhip"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])
        elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose": 
            bpart = ["lshoulder", "rshoulder", "rshoulder"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])


    if not valid_joints(part_src):
            return None

    if part_src.shape[0] == 1:
        # leg fallback
        a = part_src[0]
        b = np.float32([a[0],o_h - 1])
        part_src = np.float32([a,b])

    if part_src.shape[0] == 4:
        pass
    elif part_src.shape[0] == 3:
        # lshoulder, rshoulder, cnose
        if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            if normal[1] > 0.0:
                normal = -normal

            a = part_src[0] + normal
            b = part_src[0]
            c = part_src[1]
            d = part_src[1] + normal
            part_src = np.float32([a,b,c,d])
        else:
            assert bpart == ["lshoulder", "rshoulder", "cnose"]
            neck = 0.5*(part_src[0] + part_src[1])
            neck_to_nose = part_src[2] - neck
            part_src = np.float32([neck + 2*neck_to_nose, neck])

            # segment box
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = 1.0 / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            #part_src = np.float32([a,b,c,d])
            part_src = np.float32([b,c,d,a])
    else:
        assert part_src.shape[0] == 2

        segment = part_src[1] - part_src[0]
        normal = np.array([-segment[1],segment[0]])
        alpha = ar / 2.0
        a = part_src[0] + alpha*normal
        b = part_src[0] - alpha*normal
        c = part_src[1] - alpha*normal
        d = part_src[1] + alpha*normal
        part_src = np.float32([a,b,c,d])

    dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
    part_dst = np.float32(wh * dst)

    M = cv2.getPerspectiveTransform(part_src, part_dst)
    return M


def normalize(imgs, coords, stickmen, jo):

    out_imgs = list()
    out_stickmen = list()

    bs = len(imgs)
    for i in range(bs):
        img = imgs[i]
        joints = coords[i]
        stickman = stickmen[i]

        h,w = img.shape[:2]
        o_h = h
        o_w = w
        h = h // 4
        w = w // 4
        wh = np.array([w,h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["lshoulder","lhip","rhip","rshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["rhip", "rknee"]]
        ar = 0.5

        part_imgs = list()
        part_stickmen = list()
        for bpart in bparts:
            part_img = np.zeros((h,w,3))
            part_stickman = np.zeros((h,w,3))
            M = get_crop(bpart, joints, jo, wh, o_w, o_h, ar)

            if M is not None:
                part_img = cv2.warpPerspective(img, M, (h,w), borderMode = cv2.BORDER_REPLICATE)
                part_stickman = cv2.warpPerspective(stickman, M, (h,w), borderMode = cv2.BORDER_REPLICATE)

            part_imgs.append(part_img)
            part_stickmen.append(part_stickman)
        img = np.concatenate(part_imgs, axis = 2)
        stickman = np.concatenate(part_stickmen, axis = 2)

        """
        bpart = ["lshoulder","lhip","rhip","rshoulder"]
        dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        bpart_indices = [jo.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        img = cv2.warpPerspective(img, M, (h,w), borderMode = cv2.BORDER_REPLICATE)
        stickman = cv2.warpPerspective(stickman, M, (h,w), borderMode = cv2.BORDER_REPLICATE)
        """

        """
        # center of possible rescaling
        c = joints[jo.index("cneck")]

        # find valid body part for scale estimation
        a = joints[jo.index("lshoulder")]
        b = joints[jo.index("lhip")]
        target_length = 33.0
        if not valid_joints(a,b):
            a = joints[jo.index("rshoulder")]
            b = joints[jo.index("rhip")]
            target_length = 33.0
        if not valid_joints(a,b):
            a = joints[jo.index("rshoulder")]
            b = joints[jo.index("relbow")]
            target_length = 33.0 / 2
        if not valid_joints(a,b):
            a = joints[jo.index("lshoulder")]
            b = joints[jo.index("lelbow")]
            target_length = 33.0 / 2
        if not valid_joints(a,b):
            a = joints[jo.index("lwrist")]
            b = joints[jo.index("lelbow")]
            target_length = 33.0 / 2
        if not valid_joints(a,b):
            a = joints[jo.index("rwrist")]
            b = joints[jo.index("relbow")]
            target_length = 33.0 / 2

        if valid_joints(a,b):
            body_length = np.linalg.norm(b - a)
            factor = target_length / body_length
            img = zoom(img, factor, center = c)
            stickman = zoom(stickman, factor, center = c)
        else:
            factor = 0.25
            img = zoom(img, factor, center = c)
            stickman = zoom(stickman, factor, center = c)
        """

        out_imgs.append(img)
        out_stickmen.append(stickman)
    out_imgs = np.stack(out_imgs)
    out_stickmen = np.stack(out_stickmen)
    return out_imgs, out_stickmen


def make_mask_img(img_shape, jo, joints):
    scale_factor = img_shape[1] / 128
    masks = 3*[None]
    for i in range(3):
        masks[i] = np.zeros(img_shape[:2], dtype = "uint8")

    body = ["lhip", "lshoulder", "rshoulder", "rhip"]
    body_pts = np.array([[joints[jo.index(part),:] for part in body]], dtype = np.int32)
    cv2.fillPoly(masks[1], body_pts, 255)

    head = ["lshoulder", "chead", "rshoulder"]
    head_pts = np.array([[joints[jo.index(part),:] for part in head]], dtype = np.int32)
    cv2.fillPoly(masks[2], head_pts, 255)

    thickness = int(15 * scale_factor)
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
        r = int(11 * scale_factor)
        if r % 2 == 0:
            r = r + 1
        masks[i] = cv2.GaussianBlur(masks[i], (r,r), 0)
        maxmask = np.max(masks[i])
        if maxmask > 0:
            masks[i] = masks[i] / maxmask
    mask = np.stack(masks, axis = -1)
    mask = np.uint8(255 * mask)

    return mask



