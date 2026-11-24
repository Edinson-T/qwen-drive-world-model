"""Dependency-free reader for Waymo Open Dataset end-to-end (WOD-E2E) tfrecords.

Each tfrecord entry of the ``waymo_open_dataset_end_to_end_camera`` release is one
serialized ``waymo.open_dataset.E2EDFrame`` proto.  The official reader needs
TensorFlow plus the ``waymo-open-dataset`` package, which the released inference
environment does not ship.  This module decodes the handful of fields the scene
builder needs straight from the protobuf wire format instead, so building a
benchmark split only needs numpy.

Field numbers are taken from the upstream protos:

* ``waymo_open_dataset/protos/end_to_end_driving_data.proto`` (``E2EDFrame``,
  ``EgoTrajectoryStates``, ``EgoIntent``)
* ``waymo_open_dataset/dataset.proto`` (``Frame``, ``Context``, ``CameraImage``)

Only those fields are decoded.  Unknown fields are skipped, so a reader built for
one dataset revision keeps working as long as the fields below keep their numbers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

__all__ = [
    "CAMERA_IDS",
    "INTENT_NAMES",
    "E2EDFrame",
    "RecordMeta",
    "iter_tfrecords",
    "parse_e2ed_frame",
    "read_record",
    "scan_metadata",
]

# CameraName.Name enum values (dataset.proto).
CAMERA_IDS = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
    "REAR_LEFT": 6,
    "REAR": 7,
    "REAR_RIGHT": 8,
}

# EgoIntent.Intent enum values (end_to_end_driving_data.proto).
INTENT_NAMES = {0: "UNKNOWN", 1: "GO_STRAIGHT", 2: "GO_LEFT", 3: "GO_RIGHT"}

# E2EDFrame field numbers.
_F_FRAME = 1
_F_FUTURE = 5
_F_PAST = 6
_F_INTENT = 7
_F_PREFERENCE = 8

# dataset.proto Frame / Context / CameraImage field numbers.
_FRAME_CONTEXT = 1
_FRAME_IMAGES = 4
_CONTEXT_NAME = 1
_CAMERA_NAME = 1
_CAMERA_IMAGE = 2

# EgoTrajectoryStates field numbers.
_T_POS_X = 1
_T_POS_Y = 2
_T_VEL_X = 4
_T_VEL_Y = 5
_T_ACCEL_X = 6
_T_ACCEL_Y = 7
_T_SCORE = 8


def _varint(buf: bytes, index: int) -> tuple[int, int]:
    """Decode one base-128 varint, returning ``(value, next_index)``."""
    shift = 0
    value = 0
    while True:
        byte = buf[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield ``(field_number, wire_type, value)`` for a serialized proto.

    Values are ``int`` for varints, ``float`` for 32-bit floats and raw ``bytes``
    for 64-bit and length-delimited fields.  Groups (wire types 3/4) are not used
    by the upstream protos and raise instead of being silently mis-read.
    """
    index = 0
    size = len(buf)
    while index < size:
        key, index = _varint(buf, index)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:
            value, index = _varint(buf, index)
        elif wire_type == 1:
            value = buf[index : index + 8]
            index += 8
        elif wire_type == 2:
            length, index = _varint(buf, index)
            value = buf[index : index + length]
            index += length
        elif wire_type == 5:
            value = struct.unpack_from("<f", buf, index)[0]
            index += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _floats(buf: bytes) -> np.ndarray:
    """Decode a packed ``repeated float`` payload."""
    return np.frombuffer(buf, dtype="<f4").astype(np.float64)


def _parse_trajectory(buf: bytes) -> dict:
    """Decode an ``EgoTrajectoryStates`` into a plain dict."""
    out: dict = {}
    for number, wire_type, value in _iter_fields(buf):
        if number == _T_POS_X and wire_type == 2:
            out["pos_x"] = _floats(value)
        elif number == _T_POS_Y and wire_type == 2:
            out["pos_y"] = _floats(value)
        elif number == _T_VEL_X and wire_type == 2:
            out["vel_x"] = _floats(value)
        elif number == _T_VEL_Y and wire_type == 2:
            out["vel_y"] = _floats(value)
        elif number == _T_ACCEL_X and wire_type == 2:
            out["accel_x"] = _floats(value)
        elif number == _T_ACCEL_Y and wire_type == 2:
            out["accel_y"] = _floats(value)
        elif number == _T_SCORE and wire_type == 5:
            out["preference_score"] = float(value)
    return out


def _trajectory_score(buf: bytes) -> float | None:
    """Read only ``EgoTrajectoryStates.preference_score``."""
    for number, wire_type, value in _iter_fields(buf):
        if number == _T_SCORE and wire_type == 5:
            return float(value)
    return None


def _parse_camera(buf: bytes) -> tuple[int | None, bytes | None]:
    """Return ``(camera_id, jpeg_bytes)`` for one ``CameraImage``."""
    name = None
    image = None
    for number, wire_type, value in _iter_fields(buf):
        if number == _CAMERA_NAME and wire_type == 0:
            name = int(value)
        elif number == _CAMERA_IMAGE and wire_type == 2:
            image = value
        if name is not None and image is not None:
            break
    return name, image


def _parse_context_name(buf: bytes) -> str | None:
    """Return ``Context.name``, the run segment plus frame index."""
    for number, wire_type, value in _iter_fields(buf):
        if number == _CONTEXT_NAME and wire_type == 2:
            return value.decode("utf-8")
    return None


@dataclass
class E2EDFrame:
    """The subset of one ``E2EDFrame`` record the scene builder uses."""

    context_name: str
    scene_token: str
    frame_index: int
    images: dict[str, bytes] = field(default_factory=dict)
    future: dict = field(default_factory=dict)
    past: dict = field(default_factory=dict)
    intent: int = 0
    preferences: list[dict] = field(default_factory=list)

    @property
    def intent_name(self) -> str:
        return INTENT_NAMES.get(self.intent, "UNKNOWN")

    @property
    def preference_scores(self) -> list[float | None]:
        return [p.get("preference_score") for p in self.preferences]

    @property
    def has_labels(self) -> bool:
        """True when the frame carries at least one valid human rating."""
        return any(score is not None and score >= 0 for score in self.preference_scores)


@dataclass
class RecordMeta:
    """Indexing view of a record: identifiers plus rater scores, no images."""

    context_name: str
    scene_token: str
    frame_index: int
    preference_scores: list[float | None] = field(default_factory=list)

    @property
    def has_labels(self) -> bool:
        return any(score is not None and score >= 0 for score in self.preference_scores)


def scan_metadata(buf: bytes) -> RecordMeta | None:
    """Cheaply decode the fields needed to index a record.

    Unlike :func:`parse_e2ed_frame` this never touches the camera payloads, so a
    full-split index pass does not copy hundreds of gigabytes of JPEG bytes.
    """
    context_name = None
    scores: list[float | None] = []
    for number, wire_type, value in _iter_fields(buf):
        if number == _F_FRAME and wire_type == 2:
            for f_number, f_wire, f_value in _iter_fields(value):
                if f_number == _FRAME_CONTEXT and f_wire == 2:
                    context_name = _parse_context_name(f_value)
                    break
        elif number == _F_PREFERENCE and wire_type == 2:
            scores.append(_trajectory_score(value))

    if not context_name:
        return None
    scene_token, _, index = context_name.rpartition("-")
    if not scene_token or not index.isdigit():
        return None
    return RecordMeta(
        context_name=context_name,
        scene_token=scene_token,
        frame_index=int(index),
        preference_scores=scores,
    )


def parse_e2ed_frame(buf: bytes) -> E2EDFrame:
    """Decode one serialized ``E2EDFrame``."""
    context_name = None
    images: dict[str, bytes] = {}
    future: dict = {}
    past: dict = {}
    intent = 0
    preferences: list[dict] = []

    for number, wire_type, value in _iter_fields(buf):
        if number == _F_FRAME and wire_type == 2:
            for f_number, f_wire, f_value in _iter_fields(value):
                if f_number == _FRAME_CONTEXT and f_wire == 2:
                    context_name = _parse_context_name(f_value) or context_name
                elif f_number == _FRAME_IMAGES and f_wire == 2:
                    camera_id, jpeg = _parse_camera(f_value)
                    if camera_id is not None and jpeg is not None:
                        for camera, cid in CAMERA_IDS.items():
                            if cid == camera_id:
                                images[camera] = jpeg
                                break
        elif number == _F_FUTURE and wire_type == 2:
            future = _parse_trajectory(value)
        elif number == _F_PAST and wire_type == 2:
            past = _parse_trajectory(value)
        elif number == _F_INTENT and wire_type == 0:
            intent = int(value)
        elif number == _F_PREFERENCE and wire_type == 2:
            preferences.append(_parse_trajectory(value))

    if not context_name:
        raise ValueError("record has no frame.context.name")
    scene_token, _, index = context_name.rpartition("-")
    if not scene_token or not index.isdigit():
        raise ValueError(f"unexpected context name {context_name!r}")

    return E2EDFrame(
        context_name=context_name,
        scene_token=scene_token,
        frame_index=int(index),
        images=images,
        future=future,
        past=past,
        intent=intent,
        preferences=preferences,
    )


def iter_tfrecords(path) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(offset, length, payload)`` for every record in a tfrecord file.

    A truncated trailing record (an interrupted download or copy) is treated as
    end-of-file rather than an error.
    """
    with open(path, "rb") as handle:
        while True:
            offset = handle.tell()
            header = handle.read(8)
            if len(header) < 8:
                return
            length = struct.unpack("<Q", header)[0]
            handle.read(4)  # length CRC32
            payload = handle.read(length)
            handle.read(4)  # payload CRC32
            if len(payload) < length:
                return
            yield offset, length, payload


def read_record(path, offset: int, length: int) -> bytes:
    """Read the record at ``offset`` whose payload is ``length`` bytes."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        if len(handle.read(8)) < 8:
            raise ValueError(f"truncated record header at {offset} in {path}")
        handle.read(4)
        payload = handle.read(length)
    if len(payload) < length:
        raise ValueError(f"truncated record at {offset} in {path}")
    return payload
