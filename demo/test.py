from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_rotated_text(image, angle, xy, text, font):
    """ Draw text at an angle into an image, takes the same arguments
        as Image.text() except for:

    :param image: Image to write text into
    :param angle: Angle to write text at
    """
    width, height = image.size
    max_dim = max(width, height)
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), text, 255, font=font)
    bigger_mask = mask.resize((max_dim*8, max_dim*8),
                                resample=Image.BICUBIC)
    rotated_mask = bigger_mask.rotate(angle).resize(
        mask_size, resample=Image.LANCZOS)
    mask_xy = (max_dim - xy[0], max_dim - xy[1])
    b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
    mask = rotated_mask.crop(b_box)
    color_image = Image.new('RGBA', image.size, (255,255,255))
    image.paste(color_image, mask)

def draw_middle_line(image, angle, xy):
    width, height = image.size
    max_dim = max(width, height)
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.line([(0,0), (width, height)], fill=255, width=7)
    bigger_mask = mask.resize((max_dim*8, max_dim*8),
                                resample=Image.BICUBIC)
    rotated_mask = bigger_mask.rotate(angle).resize(
        mask_size, resample=Image.LANCZOS)
    mask_xy = (max_dim - xy[0], max_dim - xy[1])
    b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
    mask = rotated_mask.crop(b_box)
    color_image = Image.new('RGBA', image.size, (255,255,255))
    image.paste(color_image, mask)

img = np.zeros((256, 256, 3), dtype=np.uint8)
img = Image.fromarray(img)
font = ImageFont.truetype("./arial.ttf", 32)
w,h = img.size

draw_middle_line(img, -45, (100,100))
draw_rotated_text(img, -45, (150, 25), 'shape', font=font)
draw_rotated_text(img, -45, (30, 100), 'appearance', font=font)

img.show()


