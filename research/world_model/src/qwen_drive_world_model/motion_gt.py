"""Waymo Motion scenario to QD-compatible GT BEV rasters."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .perception_gt import GT_MAP_CLASS_NAMES, GT_MAP_PALETTE
from .protos import womd_pb2


QD_X_RANGE = (-30.0, 30.0)
QD_Y_RANGE = (-15.0, 15.0)
QD_RESOLUTION = 0.15
QD_HW = (200, 400)
MODEL_HW = (50, 100)

HEIGHT_BANDS = (
    (-1.0, 0.5),
    (0.5, 1.5),
    (1.5, 3.0),
    (3.0, 6.0),
)
MOTION_OBJECT_CLASS_NAMES = ("vehicle", "pedestrian", "cyclist", "other")
MOTION_TRACK_TYPE_TO_CLASS = {
    womd_pb2.Track.TYPE_VEHICLE: 0,
    womd_pb2.Track.TYPE_PEDESTRIAN: 1,
    womd_pb2.Track.TYPE_CYCLIST: 2,
    womd_pb2.Track.TYPE_OTHER: 3,
}


@dataclass(frozen=True)
class MotionMapFeature:
    kind: int
    subtype: int
    points: np.ndarray


def _map_points(points) -> np.ndarray:
    return np.asarray(
        [[point.x, point.y, getattr(point, "z", 0.0)] for point in points],
        dtype=np.float64,
    )


def scenario_map_features(scenario: womd_pb2.Scenario) -> list[MotionMapFeature]:
    features = []
    for feature in scenario.map_features:
        if feature.HasField("lane"):
            features.append(MotionMapFeature(3, int(feature.lane.type), _map_points(feature.lane.polyline)))
        elif feature.HasField("road_line"):
            features.append(
                MotionMapFeature(4, int(feature.road_line.type), _map_points(feature.road_line.polyline))
            )
        elif feature.HasField("road_edge"):
            features.append(
                MotionMapFeature(5, int(feature.road_edge.type), _map_points(feature.road_edge.polyline))
            )
        elif feature.HasField("stop_sign"):
            features.append(
                MotionMapFeature(7, 0, _map_points([feature.stop_sign.position]))
            )
        elif feature.HasField("crosswalk"):
            features.append(
                MotionMapFeature(8, 0, _map_points(feature.crosswalk.polygon))
            )
        elif feature.HasField("speed_bump"):
            features.append(
                MotionMapFeature(9, 0, _map_points(feature.speed_bump.polygon))
            )
        elif feature.HasField("driveway"):
            features.append(
                MotionMapFeature(10, 0, _map_points(feature.driveway.polygon))
            )
    return features


def ego_pose_from_state(state: womd_pb2.ObjectState) -> np.ndarray:
    return np.asarray(
        [state.center_x, state.center_y, state.heading],
        dtype=np.float64,
    )


def global_to_ego(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform global XYZ points into an ego frame with +x forward, +y left."""

    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return points.reshape(-1, 3)
    cx, cy, heading = np.asarray(pose, dtype=np.float64)
    cosine, sine = math.cos(heading), math.sin(heading)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    return np.column_stack(
        [
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            points[:, 2],
        ]
    )


def wrap_angle(angle: np.ndarray | float):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def xy_to_pixels(points: np.ndarray) -> np.ndarray:
    """QD map layout: columns span x[-30,30], rows span y[-15,15]."""

    points = np.asarray(points, dtype=np.float64)
    columns = (points[:, 0] - QD_X_RANGE[0]) / QD_RESOLUTION
    rows = (points[:, 1] - QD_Y_RANGE[0]) / QD_RESOLUTION
    return np.stack([columns, rows], axis=1)


def rasterize_map_features(
    features: list[MotionMapFeature],
    ego_pose: np.ndarray,
) -> np.ndarray:
    canvas = Image.new("L", (QD_HW[1], QD_HW[0]), color=0)
    draw = ImageDraw.Draw(canvas)
    transformed = []
    for feature in features:
        points = global_to_ego(feature.points, ego_pose)
        if not len(points):
            continue
        transformed.append((feature, xy_to_pixels(points)))

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
            for column, row in coords:
                draw.ellipse((column - 2, row - 2, column + 2, row + 2), fill=5)
    return np.asarray(canvas, dtype=np.uint8)


def track_boxes(
    scenario: womd_pb2.Scenario,
    step: int,
    ego_pose: np.ndarray,
    *,
    ego_center_z: float = 0.0,
    max_boxes: int = 64,
    min_z: float = -2.0,
    max_z: float = 7.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = []
    labels = []
    track_ids = []
    for track in scenario.tracks:
        if not 0 <= step < len(track.states):
            continue
        state = track.states[step]
        class_id = MOTION_TRACK_TYPE_TO_CLASS.get(int(track.object_type))
        if class_id is None or not state.valid:
            continue
        center = global_to_ego(
            np.asarray([[state.center_x, state.center_y, state.center_z]], dtype=np.float64),
            ego_pose,
        )[0]
        center[2] -= ego_center_z
        if not (QD_X_RANGE[0] <= center[0] <= QD_X_RANGE[1]):
            continue
        if not (QD_Y_RANGE[0] <= center[1] <= QD_Y_RANGE[1]):
            continue
        if not (min_z <= center[2] <= max_z):
            continue
        boxes.append(
            [
                center[0],
                center[1],
                center[2],
                float(state.length),
                float(state.width),
                float(state.height),
                float(wrap_angle(float(state.heading) - ego_pose[2])),
                float(state.velocity_x),
                float(state.velocity_y),
            ]
        )
        labels.append(class_id)
        track_ids.append(int(track.id))
    if len(boxes) > max_boxes:
        keep = np.argsort(np.linalg.norm(np.asarray(boxes)[:, :2], axis=1))[:max_boxes]
        boxes = [boxes[index] for index in keep]
        labels = [labels[index] for index in keep]
        track_ids = [track_ids[index] for index in keep]
    padded = np.zeros((max_boxes, 9), dtype=np.float32)
    padded_labels = np.full(max_boxes, -1, dtype=np.int8)
    padded_ids = np.full(max_boxes, -1, dtype=np.int32)
    if boxes:
        padded[: len(boxes)] = np.asarray(boxes, dtype=np.float32)
        padded_labels[: len(labels)] = np.asarray(labels, dtype=np.int8)
        padded_ids[: len(track_ids)] = np.asarray(track_ids, dtype=np.int32)
    return padded, padded_labels, padded_ids


def _box_footprint(box: np.ndarray) -> np.ndarray:
    x, y, _, length, width, _, yaw = box[:7]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local = np.asarray(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    return np.asarray(
        [[x + cosine * px - sine * py, y + sine * px + cosine * py] for px, py in local]
    )


def rasterize_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    occupancy_canvas = Image.new("L", (QD_HW[1], QD_HW[0]), color=0)
    occupancy_draw = ImageDraw.Draw(occupancy_canvas)
    height_stack = np.zeros((len(HEIGHT_BANDS), *QD_HW), dtype=np.uint8)
    for box, label in zip(boxes, labels, strict=True):
        if label < 0:
            continue
        pixels = xy_to_pixels(_box_footprint(box))
        points = [tuple(map(float, point)) for point in pixels]
        occupancy_draw.polygon(points, fill=1)
        z_min = float(box[2] - box[5] / 2.0)
        z_max = float(box[2] + box[5] / 2.0)
        for band_index, (band_min, band_max) in enumerate(HEIGHT_BANDS):
            if z_min < band_max and z_max >= band_min:
                image = Image.fromarray(height_stack[band_index])
                ImageDraw.Draw(image).polygon(points, fill=1)
                height_stack[band_index] = np.asarray(image, dtype=np.uint8)
    return np.asarray(occupancy_canvas, dtype=np.uint8), height_stack


def compress_gt_rasters(
    map_label: np.ndarray,
    occupancy: np.ndarray,
    height_slices: np.ndarray,
    output_hw: tuple[int, int] = MODEL_HW,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 13-channel WM input and 9-channel future target tensors."""

    out_h, out_w = output_hw
    in_h, in_w = map_label.shape
    block_h, block_w = in_h // out_h, in_w // out_w
    one_hot = np.eye(len(GT_MAP_CLASS_NAMES), dtype=np.float32)[map_label]
    one_hot = one_hot.reshape(out_h, block_h, out_w, block_w, -1).max(axis=(1, 3))
    importance = np.asarray([0.5, 1.0, 1.1, 1.1, 1.2, 1.3, 1.2, 1.1], np.float32)
    compact_map = (one_hot * importance).argmax(axis=-1)
    compact_occ = occupancy.reshape(out_h, block_h, out_w, block_w).max(axis=(1, 3))
    compact_height = height_slices.reshape(
        len(HEIGHT_BANDS), out_h, block_h, out_w, block_w
    ).max(axis=(2, 4))
    target = np.concatenate(
        [np.eye(len(GT_MAP_CLASS_NAMES), dtype=np.float32)[compact_map].transpose(2, 0, 1), compact_occ[None]],
        axis=0,
    )
    wm_input = np.concatenate([target, compact_height], axis=0)
    return wm_input.astype(np.float16), target.astype(np.float16)

