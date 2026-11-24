"""Waymo Perception LiDAR labels and map features as ego-frame BEV targets."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


GT_MAP_CLASS_NAMES = (
    "background",
    "lane_center",
    "road_line",
    "road_edge",
    "crosswalk",
    "stop_sign",
    "speed_bump",
    "driveway",
)
GT_MAP_PALETTE = (
    (250, 250, 250),
    (185, 198, 210),
    (255, 193, 7),
    (225, 95, 65),
    (75, 180, 170),
    (156, 39, 176),
    (255, 112, 67),
    (139, 195, 74),
)

OBJECT_CLASS_NAMES = ("vehicle", "pedestrian", "cyclist", "sign")
WAYMO_LABEL_TYPE_TO_CLASS = {
    1: 0,  # TYPE_VEHICLE
    2: 1,  # TYPE_PEDESTRIAN
    3: 3,  # TYPE_SIGN
    4: 2,  # TYPE_CYCLIST
}

BEV_X_RANGE = (0.0, 30.0)
BEV_Y_RANGE = (-15.0, 15.0)
BEV_RESOLUTION = 0.15
BEV_HW = (200, 400)


@dataclass(frozen=True)
class MapFeatureData:
    """One Waymo ``MapFeature`` decoded into points in the global map frame."""

    kind: int
    subtype: int
    points: np.ndarray


@dataclass
class PerceptionGtFrame:
    """The non-image GT payload needed to build a dynamic BEV world-model pair."""

    context_name: str
    timestamp: int
    ego_to_global: np.ndarray
    boxes: np.ndarray
    box_labels: np.ndarray
    box_ids: list[str]
    map_features: list[MapFeatureData]


def _double(wire_type: int, value) -> float:
    if wire_type != 1 or len(value) != 8:
        raise ValueError("expected a double field")
    return struct.unpack("<d", value)[0]


def _parse_map_point(buf: bytes) -> np.ndarray:
    values = {}
    for number, wire_type, value in _fields(buf):
        if number in (1, 2, 3) and wire_type == 1:
            values[number] = _double(wire_type, value)
    return np.asarray(
        [values.get(1, 0.0), values.get(2, 0.0), values.get(3, 0.0)],
        dtype=np.float64,
    )


def _parse_map_feature(buf: bytes) -> MapFeatureData | None:
    kind = 0
    subtype = 0
    points: list[np.ndarray] = []
    for number, wire_type, value in _fields(buf):
        if number == 1 and wire_type == 0:
            continue
        if number not in (3, 4, 5, 7, 8, 9, 10) or wire_type != 2:
            continue
        kind = number
        point_field = 8 if number == 3 else 2
        for child_number, child_wire, child in _fields(value):
            if child_number == 1 and child_wire == 0 and number in (4, 5):
                subtype = int(child)
            elif child_number == point_field and child_wire == 2:
                points.append(_parse_map_point(child))
        break
    if not kind or not points:
        return None
    return MapFeatureData(kind, subtype, np.stack(points))


def _parse_laser_label(
    buf: bytes,
) -> tuple[np.ndarray, int, str] | None:
    center = [0.0, 0.0, 0.0]
    dimensions = [0.0, 0.0, 0.0]
    heading = 0.0
    velocity = [0.0, 0.0]
    label_type = 0
    object_id = ""
    for number, wire_type, value in _fields(buf):
        if number == 1 and wire_type == 2:
            for child_number, child_wire, child in _fields(value):
                if child_wire != 1:
                    continue
                scalar = _double(child_wire, child)
                if child_number in (1, 2, 3):
                    center[child_number - 1] = scalar
                elif child_number == 4:
                    dimensions[1] = scalar  # width
                elif child_number == 5:
                    dimensions[0] = scalar  # length
                elif child_number == 6:
                    dimensions[2] = scalar  # height
                elif child_number == 7:
                    heading = scalar
        elif number == 2 and wire_type == 2:
            for child_number, child_wire, child in _fields(value):
                if child_number in (1, 2) and child_wire == 5:
                    velocity[child_number - 1] = struct.unpack("<f", child)[0]
        elif number == 3 and wire_type == 0:
            label_type = int(value)
        elif number == 4 and wire_type == 2:
            object_id = value.decode("utf-8", errors="replace")
    class_id = WAYMO_LABEL_TYPE_TO_CLASS.get(label_type)
    if class_id is None:
        return None
    box = np.asarray(
        [
            *center,
            *dimensions,
            heading,
            *velocity,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
        return None
    return box, class_id, object_id


def _parse_transform(buf: bytes) -> np.ndarray:
    values = [
        _double(wire_type, value)
        for number, wire_type, value in _fields(buf)
        if number == 1 and wire_type == 1
    ]
    if len(values) != 16:
        raise ValueError(f"pose transform has {len(values)} values, expected 16")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _parse_context_name(buf: bytes) -> str:
    for number, wire_type, value in _fields(buf):
        if number == 1 and wire_type == 2:
            return value.decode("utf-8", errors="replace")
    raise ValueError("Waymo Perception context has no name")


def parse_gt_frame(payload: bytes) -> PerceptionGtFrame:
    """Decode pose, 3D LiDAR labels and map features from a WOD Perception frame."""

    context_name = ""
    timestamp = 0
    ego_to_global = np.eye(4, dtype=np.float64)
    boxes = []
    box_labels = []
    box_ids = []
    map_features = []
    for number, wire_type, value in _fields(payload):
        if number == 1 and wire_type == 2:
            context_name = _parse_context_name(value)
        elif number == 2 and wire_type == 0:
            timestamp = int(value)
        elif number == 3 and wire_type == 2:
            ego_to_global = _parse_transform(value)
        elif number == 6 and wire_type == 2:
            parsed = _parse_laser_label(value)
            if parsed is not None:
                box, label, object_id = parsed
                boxes.append(box)
                box_labels.append(label)
                box_ids.append(object_id)
        elif number == 10 and wire_type == 2:
            feature = _parse_map_feature(value)
            if feature is not None:
                map_features.append(feature)
    if not context_name:
        raise ValueError("Waymo Perception frame has no context name")
    return PerceptionGtFrame(
        context_name=context_name,
        timestamp=timestamp,
        ego_to_global=ego_to_global,
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 9),
        box_labels=np.asarray(box_labels, dtype=np.int8),
        box_ids=box_ids,
        map_features=map_features,
    )


def _xy_to_pixels(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    columns = (BEV_Y_RANGE[1] - points[:, 1]) / BEV_RESOLUTION
    rows = (BEV_X_RANGE[1] - points[:, 0]) / BEV_RESOLUTION
    return np.stack([columns, rows], axis=1)


def _global_to_ego(points: np.ndarray, ego_to_global: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    global_to_ego = np.linalg.inv(np.asarray(ego_to_global, dtype=np.float64))
    return points @ global_to_ego[:3, :3].T + global_to_ego[:3, 3]


def rasterize_map_features(
    features: Iterable[MapFeatureData],
    ego_to_global: np.ndarray,
) -> np.ndarray:
    """Rasterize exact Waymo map-feature types into a forward 30 m BEV crop."""

    canvas = Image.new("L", (BEV_HW[1], BEV_HW[0]), color=0)
    draw = ImageDraw.Draw(canvas)
    transformed = []
    for feature in features:
        local = _global_to_ego(feature.points, ego_to_global)
        if not len(local):
            continue
        pixels = _xy_to_pixels(local)
        transformed.append((feature, pixels))

    # Area-like features first, then high-value linear features on top.
    draw_order = {10: 7, 9: 6, 3: 1, 4: 2, 5: 3, 8: 4}
    transformed.sort(key=lambda item: draw_order.get(item[0].kind, 0))
    for feature, pixels in transformed:
        coords = [tuple(map(float, point)) for point in pixels]
        if feature.kind in (8, 9, 10):
            draw.polygon(coords, fill=draw_order[feature.kind])
        elif feature.kind in (4, 5):
            draw.line(coords, fill=draw_order[feature.kind], width=2, joint="curve")
        elif feature.kind == 3:
            draw.line(coords, fill=1, width=2, joint="curve")
        elif feature.kind == 7:
            for x, y in coords:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=5)
    return np.asarray(canvas, dtype=np.uint8)


def _box_footprint(box: np.ndarray) -> np.ndarray:
    center_x, center_y, _, length, width, _, heading = box[:7]
    cosine, sine = np.cos(heading), np.sin(heading)
    local = np.asarray(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float64,
    )
    return np.asarray(
        [
            [
                center_x + cosine * x - sine * y,
                center_y + sine * x + cosine * y,
            ]
            for x, y in local
        ]
    )


def select_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
    *,
    x_range: tuple[float, float] = (-5.0, 40.0),
    y_range: tuple[float, float] = (-20.0, 20.0),
    max_boxes: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep labelled objects near the ego BEV and cap the variable-length set."""

    if not len(boxes):
        return np.zeros((max_boxes, 9), np.float32), np.full(max_boxes, -1, np.int8)
    keep = np.flatnonzero(
        (boxes[:, 0] >= x_range[0])
        & (boxes[:, 0] <= x_range[1])
        & (boxes[:, 1] >= y_range[0])
        & (boxes[:, 1] <= y_range[1])
    )
    if len(keep) > max_boxes:
        distance = np.linalg.norm(boxes[keep, :2], axis=1)
        keep = keep[np.argsort(distance)[:max_boxes]]
    selected = boxes[keep].astype(np.float32, copy=True)
    selected_labels = labels[keep].astype(np.int8, copy=True)
    padded = np.zeros((max_boxes, 9), np.float32)
    padded_labels = np.full(max_boxes, -1, np.int8)
    padded[: len(selected)] = selected
    padded_labels[: len(selected_labels)] = selected_labels
    return padded, padded_labels


def rasterize_boxes(boxes: np.ndarray) -> np.ndarray:
    """Rasterize 3D LiDAR-label footprints into an occupancy mask."""

    canvas = Image.new("L", (BEV_HW[1], BEV_HW[0]), color=0)
    draw = ImageDraw.Draw(canvas)
    for box in np.asarray(boxes, dtype=np.float64):
        pixels = _xy_to_pixels(_box_footprint(box))
        draw.polygon([tuple(map(float, point)) for point in pixels], fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def downsample_gt_bev(
    map_labels: np.ndarray,
    occupancy: np.ndarray,
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Compress GT raster to model input channels without dropping thin features."""

    map_labels = np.asarray(map_labels, dtype=np.int64)
    occupancy = np.asarray(occupancy, dtype=np.uint8)
    out_h, out_w = output_hw
    in_h, in_w = map_labels.shape
    if in_h % out_h or in_w % out_w:
        raise ValueError(
            f"GT BEV shape {map_labels.shape} is not divisible by target {output_hw}"
        )
    block_h, block_w = in_h // out_h, in_w // out_w
    one_hot = np.eye(len(GT_MAP_CLASS_NAMES), dtype=np.float32)[map_labels]
    one_hot = one_hot.reshape(out_h, block_h, out_w, block_w, -1).max(axis=(1, 3))
    # Preserve thin annotated lines when a 4x4 block also contains background.
    importance = np.asarray([0.5, 1.0, 1.1, 1.1, 1.2, 1.3, 1.2, 1.1], np.float32)
    compact_map = (one_hot * importance).argmax(axis=-1)
    compact_occ = occupancy.reshape(out_h, block_h, out_w, block_w).max(axis=(1, 3))
    compact_one_hot = np.eye(len(GT_MAP_CLASS_NAMES), dtype=np.float32)[compact_map]
    return np.concatenate(
        [compact_one_hot.transpose(2, 0, 1), compact_occ[None]],
        axis=0,
    ).astype(np.float16)


def _fields(buf: bytes):
    """Yield protobuf fields; copied here so world-model data has no TF dependency."""

    index = 0
    size = len(buf)
    while index < size:
        key = 0
        shift = 0
        while True:
            byte = buf[index]
            index += 1
            key |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        number, wire = key >> 3, key & 7
        if wire == 0:
            value = 0
            shift = 0
            while True:
                byte = buf[index]
                index += 1
                value |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    break
                shift += 7
        elif wire == 1:
            value = buf[index : index + 8]
            index += 8
        elif wire == 2:
            length = 0
            shift = 0
            while True:
                byte = buf[index]
                index += 1
                length |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    break
                shift += 7
            value = buf[index : index + length]
            index += length
        elif wire == 5:
            value = buf[index : index + 4]
            index += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value

