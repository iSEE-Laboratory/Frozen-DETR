"""
Modified from https://github.com/google-research/ssl_detection/blob/master/detection/utils/augmentation.py.
"""
import copy

import cv2
import mmcv
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from mmcv.image.colorspace import bgr2rgb, rgb2bgr
from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose as BaseCompose
from mmdet.datasets.pipelines import transforms


PARAMETER_MAX = 10


def int_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return int(level * maxval / max_level)


def float_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return float(level) * maxval / max_level


class RandAug(object):
    """refer to https://github.com/google-research/ssl_detection/blob/00d52272f
    61b56eade8d5ace18213cba6c74f6d8/detection/utils/augmentation.py#L240."""

    def __init__(
        self,
        prob: float = 1.0,
        magnitude: int = 10,
        random_magnitude: bool = True,
        record: bool = False,
        magnitude_limit: int = 10,
    ):
        assert 0 <= prob <= 1, f"probability should be in (0,1) but get {prob}"
        assert (
            magnitude <= PARAMETER_MAX
        ), f"magnitude should be small than max value {PARAMETER_MAX} but get {magnitude}"

        self.prob = prob
        self.magnitude = magnitude
        self.magnitude_limit = magnitude_limit
        self.random_magnitude = random_magnitude
        self.record = record
        self.buffer = None

    def __call__(self, results):
        if np.random.random() < self.prob:
            magnitude = self.magnitude
            if self.random_magnitude:
                magnitude = np.random.randint(1, magnitude)
            if self.record:
                if "aug_info" not in results:
                    results["aug_info"] = []
                results["aug_info"].append(self.get_aug_info(magnitude=magnitude))
            results = self.apply(results, magnitude)
        # clear buffer
        return results

    def apply(self, results, magnitude: int = None):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob},magnitude={self.magnitude},max_magnitude={self.magnitude_limit},random_magnitude={self.random_magnitude})"

    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                prob=1.0,
                random_magnitude=False,
                record=False,
                magnitude=self.magnitude,
            )
        )
        aug_info.update(kwargs)
        return aug_info

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class Identity(RandAug):
    def apply(self, results, magnitude: int = None):
        return results


@PIPELINES.register_module()
class AutoContrast(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            results[key] = rgb2bgr(
                np.asarray(ImageOps.autocontrast(Image.fromarray(img)), dtype=img.dtype)
            )
        return results


@PIPELINES.register_module()
class RandEqualize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            results[key] = rgb2bgr(
                np.asarray(ImageOps.equalize(Image.fromarray(img)), dtype=img.dtype)
            )
        return results


@PIPELINES.register_module()
class RandSolarize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = mmcv.solarize(
                img, min(int_parameter(magnitude, 256, self.magnitude_limit), 255)
            )
        return results


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of
    PIL."""

    def impl(pil_img, level, max_level=None):
        v = float_parameter(level, 1.8, max_level) + 0.1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


class RandEnhance(RandAug):
    op = None

    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])

            results[key] = rgb2bgr(
                np.asarray(
                    _enhancer_impl(self.op)(
                        Image.fromarray(img), magnitude, self.magnitude_limit
                    ),
                    dtype=img.dtype,
                )
            )
        return results


@PIPELINES.register_module()
class RandColor(RandEnhance):
    op = ImageEnhance.Color


@PIPELINES.register_module()
class RandContrast(RandEnhance):
    op = ImageEnhance.Contrast


@PIPELINES.register_module()
class RandBrightness(RandEnhance):
    op = ImageEnhance.Brightness


@PIPELINES.register_module()
class RandSharpness(RandEnhance):
    op = ImageEnhance.Sharpness


@PIPELINES.register_module()
class RandPosterize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            magnitude = int_parameter(magnitude, 4, self.magnitude_limit)
            results[key] = rgb2bgr(
                np.asarray(
                    ImageOps.posterize(Image.fromarray(img), 4 - magnitude),
                    dtype=img.dtype,
                )
            )
        return results


@PIPELINES.register_module()
class Sequential(BaseCompose):
    def __init__(self, transforms, record: bool = False):
        super().__init__(transforms)
        self.record = record
        self.enable_record(record)

    def enable_record(self, mode: bool = True):
        # enable children to record
        self.record = mode
        for transform in self.transforms:
            transform.enable_record(mode)


@PIPELINES.register_module()
class OneOf(Sequential):
    def __init__(self, transforms, record: bool = False):
        self.transforms = []
        for trans in transforms:
            if isinstance(trans, list):
                self.transforms.append(Sequential(trans))
            else:
                assert isinstance(trans, dict)
                self.transforms.append(Sequential([trans]))
        self.enable_record(record)

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        return transform(results)


@PIPELINES.register_module()
class ShuffledSequential(Sequential):
    def __call__(self, data):
        order = np.random.permutation(len(self.transforms))
        for idx in order:
            t = self.transforms[idx]
            data = t(data)
            if data is None:
                return None
        return data
