"""Minimal, dependency-free reader for Waymo Open Dataset Perception tfrecords.

Only what the geometry head needs is decoded:

* ``Frame.context.name`` and ``timestamp_micros``;
* ``Frame.images`` (camera JPEGs);
* ``Frame.lasers[TOP].ri_return1`` -> range image + camera projections.

The range image already carries Waymo's own per-cell camera projection
(``[H, W, 6]``: camera id, u, v, then a second projection), so no extrinsic
maths is required and no convention can be misread.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = ["PerceptionFrame", "iter_tfrecords", "parse_frame", "depth_for_camera"]

CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
TOP_LASER = 1


def iter_tfrecords(path: str | Path) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return
            length = struct.unpack("<Q", header)[0]
            handle.read(4)
            payload = handle.read(length)
            handle.read(4)
            if len(payload) < length:
                return
            yield payload


def _varint(buf: bytes, index: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        byte = buf[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7


def _fields(buf: bytes):
    index = 0
    size = len(buf)
    while index < size:
        key, index = _varint(buf, index)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, index = _varint(buf, index)
        elif wire == 1:
            value = buf[index : index + 8]
            index += 8
        elif wire == 2:
            length, index = _varint(buf, index)
            value = buf[index : index + length]
            index += length
        elif wire == 5:
            value = struct.unpack_from("<f", buf, index)[0]
            index += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield number, wire, value


def _packed_int32(buf: bytes) -> np.ndarray:
    out = []
    index = 0
    size = len(buf)
    while index < size:
        shift = 0
        value = 0
        while True:
            byte = buf[index]
            index += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        out.append(value)
    return np.asarray(out, dtype=np.int64)


def _matrix_float(buf: bytes) -> tuple[list[int], np.ndarray]:
    dims: list[int] = []
    chunks: list[np.ndarray] = []
    for number, wire, value in _fields(buf):
        if number == 1 and wire == 2:
            chunks.append(np.frombuffer(value, dtype="<f4"))
        elif number == 2 and wire == 2:
            dims = [v for n, w, v in _fields(value) if n == 1 and w == 0]
    return dims, np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


@dataclass
class PerceptionFrame:
    context_name: str
    timestamp: int
    cameras: dict[int, bytes]
    range_image: np.ndarray  # [H, W, 4], channel 0 = range
    projection: np.ndarray  # [H, W, 6], channel 0 = camera id, 1/2 = u/v

    @property
    def segment(self) -> str:
        return self.context_name.split("-")[0]

    @property
    def frame_index(self) -> int:
        tail = self.context_name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1


def parse_frame(buf: bytes) -> PerceptionFrame:
    context = None
    timestamp = 0
    images = []
    lasers = []
    for number, wire, value in _fields(buf):
        if number == 1 and wire == 2:
            context = value
        elif number == 2 and wire == 0:
            timestamp = int(value)
        elif number == 4 and wire == 2:
            images.append(value)
        elif number == 5 and wire == 2:
            lasers.append(value)

    name = ""
    if context is not None:
        for n, w, v in _fields(context):
            if n == 1 and w == 2:
                name = v.decode("utf-8")
                break

    cameras: dict[int, bytes] = {}
    for image in images:
        cam_name, jpeg = None, None
        for n, w, v in _fields(image):
            if n == 1 and w == 0:
                cam_name = int(v)
            elif n == 2 and w == 2:
                jpeg = v
        if cam_name is not None and jpeg is not None:
            cameras[cam_name] = jpeg

    range_image = np.zeros((0, 0, 4), dtype=np.float32)
    projection = np.zeros((0, 0, 6), dtype=np.int64)
    for laser in lasers:
        fields = list(_fields(laser))
        if int(next(v for n, w, v in fields if n == 1 and w == 0)) != TOP_LASER:
            continue
        first_return = next(v for n, w, v in fields if n == 2 and w == 2)
        rf = list(_fields(first_return))
        range_dims, range_data = _matrix_float(zlib.decompress(next(v for n, w, v in rf if n == 2 and w == 2)))
        proj_message = zlib.decompress(next(v for n, w, v in rf if n == 3 and w == 2))
        proj_data = _packed_int32(next(v for n, w, v in _fields(proj_message) if n == 1 and w == 2))
        if len(range_dims) == 3:
            h, w = int(range_dims[0]), int(range_dims[1])
            range_channels = int(range_dims[2])
            range_expected = h * w * range_channels
            proj_expected = h * w * 6
            if range_data.size < range_expected:
                range_data = np.pad(range_data, (0, range_expected - range_data.size))
            else:
                range_data = range_data[:range_expected]
            # The packed stream occasionally carries a few trailing bytes; keep
            # exactly the declared shape.
            if proj_data.size < proj_expected:
                proj_data = np.pad(proj_data, (0, proj_expected - proj_data.size))
            else:
                proj_data = proj_data[:proj_expected]
            range_image = range_data.reshape(h, w, range_channels)
            projection = proj_data.reshape(h, w, 6)
        break

    return PerceptionFrame(
        context_name=name,
        timestamp=timestamp,
        cameras=cameras,
        range_image=range_image,
        projection=projection,
    )


def depth_for_camera(
    frame: PerceptionFrame,
    camera_id: int,
    image_size: tuple[int, int],
    out_size: tuple[int, int],
    min_depth: float = 1.0,
    max_depth: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer the cells projected into ``camera_id`` into a depth map.

    ``image_size`` is the native ``(width, height)``; ``out_size`` the target grid.
    """
    native_w, native_h = int(image_size[0]), int(image_size[1])
    out_w, out_h = int(out_size[0]), int(out_size[1])
    range_image, projection = frame.range_image, frame.projection
    if range_image.size == 0:
        return np.zeros((out_h, out_w), np.float32), np.zeros((out_h, out_w), bool)

    depth_values = range_image[..., 0]
    selected = (projection[..., 0] == camera_id) & (depth_values > min_depth) & (depth_values < max_depth)
    u = projection[..., 1][selected]
    v = projection[..., 2][selected]
    d = depth_values[selected]

    ui = np.clip((u * (out_w / native_w)).astype(np.int64), 0, out_w - 1)
    vi = np.clip((v * (out_h / native_h)).astype(np.int64), 0, out_h - 1)
    depth = np.full((out_h, out_w), np.inf, dtype=np.float32)
    np.minimum.at(depth, (vi, ui), d.astype(np.float32))
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return depth, mask
