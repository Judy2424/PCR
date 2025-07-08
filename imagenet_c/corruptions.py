# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

# /////////////// Corruption Helpers ///////////////
import numba
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
# from wand.image import Image as WandImage
# from wand.api import library as wandlibrary
# import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from pkg_resources import resource_filename
from scipy import interpolate
from scipy.ndimage import convolve
warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
# wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
#                                               ctypes.c_double,  # radius
#                                               ctypes.c_double,  # sigma
#                                               ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
# class MotionImage(WandImage):
#     def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
#         wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

class MotionImage:
    def __init__(self, blob=None, img_array=None, img_path=None):
        if blob is not None:
            file_bytes = np.frombuffer(blob, dtype=np.uint8)
            self.image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif img_array is not None:
            self.image = img_array
        elif img_path is not None:
            self.image = cv2.imread(img_path)
        else:
            raise ValueError("Either 'blob', 'img_array', or 'img_path' must be provided.")

    def motion_blur(self, radius=5, sigma=0, angle=0):
        kernel_size = max(3, int(2 * radius + 1))  # radius → 커널 크기 변환
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2] = np.ones(kernel_size)
        kernel /= kernel_size

        rot_mat = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))

        self.image = cv2.filter2D(self.image, -1, kernel)
        return self.image

    def make_blob(self, format='.jpg'):
        """이미지 데이터를 blob (바이트 스트림) 으로 변환"""
        success, buffer = cv2.imencode(format, self.image)
        if not success:
            raise RuntimeError("Image encoding failed")
        return buffer.tobytes()

    @staticmethod
    def from_blob(blob):
        """blob 데이터를 사용해 MotionImage 인스턴스 생성"""
        return MotionImage(blob=blob)
# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////
interpolation_function_dict = dict()
def gaussian_noise(x, severity=1):
    if "gaussian_noise" not in interpolation_function_dict:
        interpolation_function_dict["gaussian_noise"] = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [0, 0.08, 0.12, 0.18, 0.26, 0.38],
            axis=0, kind="linear"
        )
    c = interpolation_function_dict["gaussian_noise"](severity)
    x_arr = np.array(x) / 255.0
    noise = np.random.normal(scale=c, size=x_arr.shape)
    out = np.clip(x_arr + noise, 0, 1) * 255
    return out.astype(np.uint8)



def shot_noise(x, severity=1):
    if "shot noise" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                0,
                float(1) / 60,
                float(1) / 25,
                float(1) / 12,
                float(1) / 5,
                float(1) / 3,
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["shot noise"] = f

    f = interpolation_function_dict["shot noise"]

    c = f(severity)
    if c != 0:
        c = float(1) / c
    else:
        c = 9999
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


# def impulse_noise(x, severity=1):
#     if "impulse_noise" not in interpolation_function_dict:
#         interpolation_function_dict["impulse_noise"] = interpolate.interp1d(
#             [0, 1, 2, 3, 4, 5],
#             [0.0, 0.03, 0.06, 0.09, 0.17, 0.27],
#             axis=0, kind="linear"
#         )
#     c = interpolation_function_dict["impulse_noise"](severity)
#     x_arr = np.array(x).copy()
    
#     h, w, _ = x_arr.shape
#     # Generate (H, W) mask: 0 (pepper), 1 (salt), 2 (keep original)
#     mask = np.random.choice([0, 1, 2], size=(h, w), p=[c / 2, c / 2, 1 - c])
    
#     # Create boolean masks for pepper and salt
#     pepper_mask = mask == 0
#     salt_mask = mask == 1
    
#     # Expand to 3 channels
#     pepper_mask = np.stack([pepper_mask] * 3, axis=-1)
#     salt_mask = np.stack([salt_mask] * 3, axis=-1)
    
#     # Apply noise
#     x_arr[pepper_mask] = 0
#     x_arr[salt_mask] = 255
    
#     return x_arr.astype(np.uint8)
def impulse_noise(x, severity=1):
    if "impulse noise" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [0, 0.03, 0.06, 0.09, 0.17, 0.27], axis=0, kind="linear"
        )
        interpolation_function_dict["impulse noise"] = f

    f = interpolation_function_dict["impulse noise"]

    c = f(severity)

    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255



def glass_blur_slow(x, severity=1):
    if "glass blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (0.0, 0.0, 0.0),
                (0.7, 1, 2),
                (0.9, 2, 1),
                (1, 2, 3),
                (1.1, 3, 2),
                (1.5, 4, 2),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["glass blur"] = f

    f = interpolation_function_dict["glass blur"]

    c = f(severity)

    if c[1] < 1:
        c[1] = 1

    # x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], multichannel=True) * 255)
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis = -1) * 255)

    # locally shuffle pixels
    for i in range(round(c[2])):
        for h in range(224 - round(c[1]), round(c[1]), -1):
            for w in range(224 - round(c[1]), round(c[1]), -1):
                dx, dy = np.random.randint(-round(c[1]), round(c[1]), size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    # return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1) * 255
    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis = -1), 0, 1) * 255


@numba.jit()
def shuffle_pixels(x, c):
    # locally shuffle pixels
    for i in range(round(c[2])):
        for h in range(224 - round(c[1]), round(c[1]), -1):
            for w in range(224 - round(c[1]), round(c[1]), -1):
                dx, dy = np.random.randint(-round(c[1]), round(c[1]), size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x


def glass_blur(x, severity=1):
    if "glass blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (0.0, 0.0, 0.0),
                (0.7, 1, 2),
                (0.9, 2, 1),
                (1, 2, 3),
                (1.1, 3, 2),
                (1.5, 4, 2),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["glass blur"] = f

    f = interpolation_function_dict["glass blur"]

    c = f(severity)

    if c[1] < 1:
        c[1] = 1

    # x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], multichannel=True) * 255)
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis = -1) * 255)

    shuffle_pixels(x, c)

    # return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1) * 255
    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis = -1), 0, 1) * 255


# def defocus_blur(x, severity=1):
#     if "defocus_blur" not in interpolation_function_dict:
#         interpolation_function_dict["defocus_blur"] = interpolate.interp1d(
#             [0,1,2,3,4,5],
#             [0.0, 1.5, 2.0, 2.75, 3.25, 4.0],
#             axis=0, kind="linear"
#         )
#     c = interpolation_function_dict["defocus_blur"](severity)
#     radius = int(c)
#     x = np.array(x)
#     h, w = x.shape[:2]
#     # Create disk kernel dynamically based on radius
#     L = np.arange(-radius, radius+1)
#     X, Y = np.meshgrid(L, L)
#     disk = ((X**2 + Y**2) <= radius**2).astype(np.float32)
#     disk /= np.sum(disk)

#     x_arr = np.array(x, dtype=np.float32) / 255.0
#     x_blurred = np.stack([
#         convolve(x_arr[..., c], disk, mode='reflect') for c in range(x_arr.shape[2])
#     ], axis=2)
#     return (np.clip(x_blurred, 0, 1) * 255).astype(np.uint8)
def defocus_blur(x, severity=1):
    if "defocus blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(0.0, 0.0), (3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["defocus blur"] = f

    f = interpolation_function_dict["defocus blur"]

    c = f(severity)

    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    if "motion blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(0.0, 0.0), (10, 3), (15, 5), (15, 8), (15, 12), (20, 15)],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["motion blur"] = f

    f = interpolation_function_dict["motion blur"]

    c = f(severity)

    output = BytesIO()
    x.save(output, format="PNG")
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    if "zoom blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (1.0, 1.0, 0.01),
                (1, 1.11, 0.01),
                (1, 1.16, 0.01),
                (1, 1.21, 0.02),
                (1, 1.26, 0.02),
                (1, 1.31, 0.03),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["zoom blur"] = f

    f = interpolation_function_dict["zoom blur"]

    c = f(severity)
    c = np.arange(c[0], c[1], c[2])

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    if "fog" not in interpolation_function_dict:
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(0.0, 2.0), (1.5,2), (2,2), (2.5,1.7), (2.5,1.5), (3,1.4)],
            axis=0, kind="linear"
        )
        interpolation_function_dict["fog"] = f

    c = interpolation_function_dict["fog"](severity)
    x_arr = np.array(x, dtype=np.float32) / 255.0
    h, w = x_arr.shape[:2]
    max_val = x_arr.max()

    fractal = plasma_fractal(wibbledecay=c[1])
    from PIL import Image
    fractal_img = Image.fromarray((fractal * 255).astype(np.uint8))
    fractal_img = fractal_img.resize((w, h), Image.BILINEAR)
    fractal = np.array(fractal_img).astype(np.float32) / 255.0
    fractal = fractal[..., np.newaxis]

    x_out = np.clip(
        (x_arr * max_val) / (max_val + c[0]) + c[0] * fractal / (max_val + c[0]),
        0, 1
    ) * 255
    return x_out.astype(np.uint8)


from PIL import Image

def frost(x, severity=1):
    if "frost" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(1.0, 0.0), (1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)],
            axis=0,
            kind="linear",
        )

        interpolation_function_dict["frost"] = f
    f = interpolation_function_dict["frost"]

    c = f(severity)

    x_arr = np.array(x).astype(np.float32) / 255.0
    h, w = x_arr.shape[:2]

    idx = np.random.randint(6)  # 6개 파일이므로 0~5
    filename = [
        "./imagenet_c/frost/frost1.png",
        "./imagenet_c/frost/frost2.png",
        "./imagenet_c/frost/frost3.png",
        "./imagenet_c/frost/frost4.jpg",
        "./imagenet_c/frost/frost5.jpg",
        "./imagenet_c/frost/frost6.jpg",
    ][idx]
    filename = os.path.abspath(filename)

    frost_img = Image.open(filename).convert("RGB")  # RGBA → RGB 강제 변환
    frost_img = frost_img.resize((w, h), Image.BILINEAR)
    frost = np.array(frost_img).astype(np.float32) / 255.0

    # 혼합
    out = np.clip(c[0] * x_arr  + c[1] * frost , 0, 1) * 255
    return out.astype(np.uint8)



def snow(x, severity=1):
    # interpolate parameters
    if "snow" not in interpolation_function_dict:
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (0.1, 0.3, 3, 1.0, 10, 4, 1.0),
                (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
                (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
                (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
                (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
                (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
            ], axis=0, kind="linear"
        )
        interpolation_function_dict["snow"] = f

    c = interpolation_function_dict["snow"](severity)
    x_arr = np.array(x, dtype=np.float32) / 255.0
    h, w = x_arr.shape[:2]

    snow_layer = np.random.normal(size=(h, w), loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer = np.pad(
        snow_layer,
        [(0, max(0, h - snow_layer.shape[0])), (0, max(0, w - snow_layer.shape[1])), (0, 0)],
        mode="constant"
    )[0:h, 0:w, :]
    snow_layer[snow_layer < c[3]] = 0

    # snow_img = c[6] * x_arr + (1 - c[6]) * np.maximum(
    #     x_arr, cv2.cvtColor((x_arr*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255*1.5+0.5
    # )
    gray_layer = cv2.cvtColor((x_arr*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255*1.5 + 0.5
    gray_layer = gray_layer[..., None]  # 채널 차원 추가

    snow_img = c[6] * x_arr + (1 - c[6]) * np.maximum(x_arr, gray_layer)
    x_out = np.clip(snow_img + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    return x_out.astype(np.uint8)

# def contrast(x, severity=1):
#     c = [0.4, .3, .2, .1, .05][severity - 1]

#     x = np.array(x) / 255.
#     means = np.mean(x, axis=(0, 1), keepdims=True)
#     return np.clip((x - means) * c + means, 0, 1) * 255

def contrast(x, severity=1):
    if "contrast" not in interpolation_function_dict:
        interpolation_function_dict["contrast"] = interpolate.interp1d(
            [0,1,2,3,4,5],
            [1.0,0.4,0.3,0.2,0.1,0.05],
            axis=0, kind="linear"
        )
    c = interpolation_function_dict["contrast"](severity)
    x_arr = np.array(x) / 255.0
    means = np.mean(x_arr, axis=(0,1), keepdims=True)
    out = np.clip((x_arr - means)*c + means, 0, 1) * 255
    return out.astype(np.uint8)


def brightness(x, severity=1):
    if "brightness" not in interpolation_function_dict:
        interpolation_function_dict["brightness"] = interpolate.interp1d(
            [0,1,2,3,4,5],
            [0.0,0.1,0.2,0.3,0.4,0.5],
            axis=0, kind="linear"
        )
    c = interpolation_function_dict["brightness"](severity)
    x_arr = np.array(x) / 255.0
    # RGB -> HSV -> 밝기 조절 -> 다시 RGB
    hsv = sk.color.rgb2hsv(x_arr)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + c, 0, 1)
    out = sk.color.hsv2rgb(hsv)
    return (np.clip(out,0,1)*255).astype(np.uint8)



# def jpeg_compression(x, severity=1):
#     if "jpeg" not in interpolation_function_dict:
#         f = interpolate.interp1d(
#             [0, 1, 2, 3, 4, 5],
#             [85, 25, 18, 15, 10, 7],
#             axis=0, kind="linear"
#         )
#         interpolation_function_dict["jpeg"] = f

#     quality = int(interpolation_function_dict["jpeg"](severity).item())
#     output = BytesIO()
#     img = PILImage.fromarray(np.array(x).astype(np.uint8))
#     img.save(output, "JPEG", quality=quality)
#     output.seek(0)
#     return np.array(PILImage.open(output))
def jpeg_compression(x, severity=1):
    if "jpeg" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [85, 25, 18, 15, 10, 7], axis=0, kind="linear"
        )
        interpolation_function_dict["jpeg"] = f

    f = interpolation_function_dict["jpeg"]
    c = f(severity)

    c = round(c.item())
    output = BytesIO()

    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)
    return x


# def pixelate(x, severity=1):
#     if "pixelate" not in interpolation_function_dict:
#         interpolation_function_dict["pixelate"] = interpolate.interp1d(
#             [0,1,2,3,4,5],
#             [1.0,0.6,0.5,0.4,0.3,0.25],
#             axis=0, kind="linear"
#         )
#     c = interpolation_function_dict["pixelate"](severity)
#     x_pil = PILImage.fromarray(np.array(x).astype(np.uint8))
#     w, h = x_pil.size
#     small = x_pil.resize((int(w*c), int(h*c)), PILImage.BOX)
#     out = small.resize((w, h), PILImage.BOX)
#     return np.array(out).astype(np.uint8)
def pixelate(x, severity=1):
    if "pixelate" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [1.0, 0.6, 0.5, 0.4, 0.3, 0.25], axis=0, kind="linear"
        )
        interpolation_function_dict["pixelate"] = f

    f = interpolation_function_dict["pixelate"]
    c = f(severity)
    x_pil = PILImage.fromarray(np.array(x).astype(np.uint8))
    w, h = x_pil.size
    small = x_pil.resize((int(w*c), int(h*c)), PILImage.BOX)
    out = small.resize((w, h), PILImage.BOX)
    return np.array(out).astype(np.uint8)


from scipy.ndimage import map_coordinates
def elastic_transform(image, severity=1):
    if "elastic_transform" not in interpolation_function_dict:
        interpolation_function_dict["elastic_transform"] = interpolate.interp1d(
            [0,1,2,3,4,5],
            [
                (0,999,0),
                (488,170.8,24.4),
                (488,19.52,48.8),
                (12.2,2.44,48.8),
                (17.08,2.44,48.8),
                (29.28,2.44,48.8),
            ], axis=0, kind="linear"
        )
    c = interpolation_function_dict["elastic_transform"](severity)
    arr = np.array(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    # random affine
    center = np.float32((h//2, w//2))
    sq = min(h,w)//3
    pts1 = np.float32([center + sq, [center[0]+sq, center[1]-sq], center - sq])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    arr = cv2.warpAffine(arr, M, (w,h), borderMode=cv2.BORDER_REFLECT_101)
    # elastic offsets
    dx = gaussian(np.random.uniform(-1,1,size=(h,w)), c[1], channel_axis=None) * c[0]
    dy = gaussian(np.random.uniform(-1,1,size=(h,w)), c[1], channel_axis=None) * c[0]
    dx = dx[..., None]; dy = dy[..., None]
    # meshgrid + map_coordinates
    xg, yg = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.array([yg+dy[...,0], xg+dx[...,0]])
    for d in range(arr.shape[2]):
        arr[...,d] = map_coordinates(arr[...,d], coords, order=1, mode="reflect").reshape((h,w))
    return (np.clip(arr,0,1)*255).astype(np.uint8)

# def gaussian_noise(x, severity=1):
#     c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

#     x = np.array(x) / 255.
#     return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# def shot_noise(x, severity=1):
#     c = [60, 25, 12, 5, 3][severity - 1]

#     x = np.array(x) / 255.
#     return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


# def impulse_noise(x, severity=1):
#     c = [.03, .06, .09, 0.17, 0.27][severity - 1]

#     x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
#     return np.clip(x, 0, 1) * 255


# def speckle_noise(x, severity=1):
#     c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

#     x = np.array(x) / 255.
#     return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# def fgsm(x, source_net, severity=1):
#     c = [8, 16, 32, 64, 128][severity - 1]

#     x = V(x, requires_grad=True)
#     logits = source_net(x)
#     source_net.zero_grad()
#     loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
#     loss.backward()

#     return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))


# def gaussian_blur(x, severity=1):
#     c = [1, 2, 3, 4, 6][severity - 1]

#     x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
#     return np.clip(x, 0, 1) * 255


# def glass_blur(x, severity=1):
#     # sigma, max_delta, iterations
#     c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

#     x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

#     # locally shuffle pixels
#     for i in range(c[2]):
#         for h in range(224 - c[1], c[1], -1):
#             for w in range(224 - c[1], c[1], -1):
#                 dx, dy = np.random.randint(-c[1], c[1], size=(2,))
#                 h_prime, w_prime = h + dy, w + dx
#                 # swap
#                 x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

#     return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


# def defocus_blur(x, severity=1):
#     c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

#     x = np.array(x) / 255.
#     kernel = disk(radius=c[0], alias_blur=c[1])

#     channels = []
#     for d in range(3):
#         channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
#     channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

#     return np.clip(channels, 0, 1) * 255


# def motion_blur(x, severity=1):
#     c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

#     output = BytesIO()
#     x.save(output, format='PNG')
#     x = MotionImage(blob=output.getvalue())

#     x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)

#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


# def zoom_blur(x, severity=1):
#     c = [np.arange(1, 1.11, 0.01),
#          np.arange(1, 1.16, 0.01),
#          np.arange(1, 1.21, 0.02),
#          np.arange(1, 1.26, 0.02),
#          np.arange(1, 1.31, 0.03)][severity - 1]

#     x = (np.array(x) / 255.).astype(np.float32)
#     out = np.zeros_like(x)
#     for zoom_factor in c:
#         out += clipped_zoom(x, zoom_factor)

#     x = (x + out) / (len(c) + 1)
#     return np.clip(x, 0, 1) * 255


# def fog(x, severity=1):
#     c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

#     x = np.array(x) / 255.
#     max_val = x.max()
#     x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
#     return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


# def frost(x, severity=1):
#     c = [(1, 0.4),
#          (0.8, 0.6),
#          (0.7, 0.7),
#          (0.65, 0.7),
#          (0.6, 0.75)][severity - 1]
#     idx = np.random.randint(5)
#     filename = [resource_filename(__name__, 'frost/frost1.png'),
#                 resource_filename(__name__, 'frost/frost2.png'),
#                 resource_filename(__name__, 'frost/frost3.png'),
#                 resource_filename(__name__, 'frost/frost4.jpg'),
#                 resource_filename(__name__, 'frost/frost5.jpg'),
#                 resource_filename(__name__, 'frost/frost6.jpg')][idx]
#     frost = cv2.imread(filename)
#     # randomly crop and convert to rgb
#     x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
#     frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

#     return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


# def snow(x, severity=1):
#     c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
#          (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
#          (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
#          (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
#          (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

#     x = np.array(x, dtype=np.float32) / 255.
#     snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

#     snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
#     snow_layer[snow_layer < c[3]] = 0

#     snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
#     output = BytesIO()
#     snow_layer.save(output, format='PNG')
#     snow_layer = MotionImage(blob=output.getvalue())

#     snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

#     snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
#                               cv2.IMREAD_UNCHANGED) / 255.
#     # snow_layer = snow_layer[..., np.newaxis]

#     x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
#     return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


# def spatter(x, severity=1):
#     c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
#          (0.65, 0.3, 3, 0.68, 0.6, 0),
#          (0.65, 0.3, 2, 0.68, 0.5, 0),
#          (0.65, 0.3, 1, 0.65, 1.5, 1),
#          (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
#     x = np.array(x, dtype=np.float32) / 255.

#     liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

#     liquid_layer = gaussian(liquid_layer, sigma=c[2])
#     liquid_layer[liquid_layer < c[3]] = 0
#     if c[5] == 0:
#         liquid_layer = (liquid_layer * 255).astype(np.uint8)
#         dist = 255 - cv2.Canny(liquid_layer, 50, 150)
#         dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
#         _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
#         dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
#         dist = cv2.equalizeHist(dist)
#         ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
#         dist = cv2.filter2D(dist, cv2.CV_8U, ker)
#         dist = cv2.blur(dist, (3, 3)).astype(np.float32)

#         m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
#         m /= np.max(m, axis=(0, 1))
#         m *= c[4]

#         # water is pale turqouise
#         color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
#                                 238 / 255. * np.ones_like(m[..., :1]),
#                                 238 / 255. * np.ones_like(m[..., :1])), axis=2)

#         color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
#         x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

#         return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
#     else:
#         m = np.where(liquid_layer > c[3], 1, 0)
#         m = gaussian(m.astype(np.float32), sigma=c[4])
#         m[m < 0.8] = 0

#         # mud brown
#         color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
#                                 42 / 255. * np.ones_like(x[..., :1]),
#                                 20 / 255. * np.ones_like(x[..., :1])), axis=2)

#         color *= m[..., np.newaxis]
#         x *= (1 - m[..., np.newaxis])

#         return np.clip(x + color, 0, 1) * 255


# def contrast(x, severity=1):
#     c = [0.4, .3, .2, .1, .05][severity - 1]

#     x = np.array(x) / 255.
#     means = np.mean(x, axis=(0, 1), keepdims=True)
#     return np.clip((x - means) * c + means, 0, 1) * 255


# def brightness(x, severity=1):
#     c = [.1, .2, .3, .4, .5][severity - 1]

#     x = np.array(x) / 255.
#     x = sk.color.rgb2hsv(x)
#     x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
#     x = sk.color.hsv2rgb(x)

#     return np.clip(x, 0, 1) * 255


# def saturate(x, severity=1):
#     c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

#     x = np.array(x) / 255.
#     x = sk.color.rgb2hsv(x)
#     x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
#     x = sk.color.hsv2rgb(x)

#     return np.clip(x, 0, 1) * 255


# def jpeg_compression(x, severity=1):
#     c = [25, 18, 15, 10, 7][severity - 1]

#     output = BytesIO()
#     x.save(output, 'JPEG', quality=c)
#     x = PILImage.open(output)

#     return x


# def pixelate(x, severity=1):
#     c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

#     x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
#     x = x.resize((224, 224), PILImage.BOX)

#     return x


# # mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
# def elastic_transform(image, severity=1):
#     c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
#          (244 * 2, 244 * 0.08, 244 * 0.2),
#          (244 * 0.05, 244 * 0.01, 244 * 0.02),
#          (244 * 0.07, 244 * 0.01, 244 * 0.02),
#          (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

#     image = np.array(image, dtype=np.float32) / 255.
#     shape = image.shape
#     shape_size = shape[:2]

#     # random affine
#     center_square = np.float32(shape_size) // 2
#     square_size = min(shape_size) // 3
#     pts1 = np.float32([center_square + square_size,
#                        [center_square[0] + square_size, center_square[1] - square_size],
#                        center_square - square_size])
#     pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
#     M = cv2.getAffineTransform(pts1, pts2)
#     image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

#     dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
#                    c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
#     dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
#                    c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
#     dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

#     x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
#     indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
#     return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Corruptions ///////////////
