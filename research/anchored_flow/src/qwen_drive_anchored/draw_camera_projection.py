"""Project RL base and a Best-of-N candidate onto a calibrated camera image."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .draw_base_bestn import _rfs_for_base_and_candidates
from .paper_figures import _find_scene_record, _load_predictions


CAMERA_IDS = {"FRONT": 1, "FRONT_LEFT": 2, "FRONT_RIGHT": 3}


def _double_values(value, wire_type: int) -> list[float]:
    if wire_type == 1:
        return [float(np.frombuffer(value, dtype="<f8")[0])]
    if wire_type == 2:
        return np.frombuffer(value, dtype="<f8").tolist()
    return [float(value)]


def _parse_calibrations(payload: bytes, iterator_fields) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    calibrations = {}
    for number, wire, value in iterator_fields(payload):
        if number != 1 or wire != 2:
            continue
        for frame_number, frame_wire, frame_value in iterator_fields(value):
            if frame_number != 1 or frame_wire != 2:
                continue
            for context_number, context_wire, context_value in iterator_fields(frame_value):
                if context_number != 2 or context_wire != 2:
                    continue
                camera_id = None
                intrinsic = []
                extrinsic = []
                for field, field_wire, field_value in iterator_fields(context_value):
                    if field == 1 and field_wire == 0:
                        camera_id = int(field_value)
                    elif field == 2:
                        intrinsic.extend(_double_values(field_value, field_wire))
                    elif field == 3 and field_wire == 2:
                        for transform_field, transform_wire, transform_value in iterator_fields(field_value):
                            if transform_field == 1:
                                extrinsic.extend(
                                    _double_values(transform_value, transform_wire)
                                )
                if camera_id is not None and len(intrinsic) == 9 and len(extrinsic) == 16:
                    calibrations[camera_id] = (
                        np.asarray(intrinsic, dtype=np.float64),
                        np.asarray(extrinsic, dtype=np.float64).reshape(4, 4),
                    )
    return calibrations


def _project_to_image(
    trajectory_xy: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_vehicle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [
            trajectory_xy,
            np.zeros((len(trajectory_xy), 1), dtype=np.float64),
            np.ones((len(trajectory_xy), 1), dtype=np.float64),
        ],
        axis=1,
    )
    camera_aligned = (np.linalg.inv(camera_to_vehicle) @ points.T).T
    # Waymo calibration uses x-forward, y-left, z-up. OpenCV projection uses
    # x-right, y-down, z-forward.
    optical = np.column_stack(
        (-camera_aligned[:, 1], -camera_aligned[:, 2], camera_aligned[:, 0])
    )
    matrix = np.asarray(
        [
            [intrinsic[0], 0.0, intrinsic[2]],
            [0.0, intrinsic[1], intrinsic[3]],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion = intrinsic[4:9].reshape(1, -1)
    pixels, _ = cv2.projectPoints(
        optical,
        np.zeros(3),
        np.zeros(3),
        matrix,
        distortion,
    )
    return pixels[:, 0], optical[:, 2]


def _draw_path(
    image: np.ndarray,
    pixels: np.ndarray,
    depth: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    valid = (
        (depth > 0.25)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= -20)
        & (pixels[:, 0] < width + 20)
        & (pixels[:, 1] >= -20)
        & (pixels[:, 1] < height + 20)
    )
    for index in range(len(pixels) - 1):
        if valid[index] and valid[index + 1]:
            cv2.line(
                image,
                tuple(np.round(pixels[index]).astype(int)),
                tuple(np.round(pixels[index + 1]).astype(int)),
                color,
                7,
                cv2.LINE_AA,
            )
    visible = np.flatnonzero(valid)
    if not len(visible):
        return
    endpoint = tuple(np.round(pixels[visible[-1]]).astype(int))
    cv2.circle(image, endpoint, 9, color, -1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--camera",
        choices=tuple(CAMERA_IDS),
        default="FRONT_LEFT",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "waymo_e2e"))
    import e2e_proto

    base_records = _load_predictions(args.baseline)
    candidate_records = _load_predictions(args.candidates)
    if args.token not in base_records or args.token not in candidate_records:
        raise SystemExit(f"token not found: {args.token}")
    base_record = base_records[args.token]
    candidate_record = candidate_records[args.token]
    scene_record = _find_scene_record(args.scenes, args.token)
    base_rfs, candidate_rfs = _rfs_for_base_and_candidates(
        base_record,
        candidate_record,
    )
    candidate_index = int(np.argmax(candidate_rfs))

    with gzip.open(args.index, "rt") as handle:
        index = json.load(handle)
    scene_token, frame = args.token.rsplit("-", 1)
    shard_id, offset, length = index["index"][scene_token][frame]
    shard = args.raw_root / index["shards"][shard_id]
    payload = e2e_proto.read_record(shard, int(offset), int(length))
    calibrations = _parse_calibrations(payload, e2e_proto._iter_fields)
    camera_id = CAMERA_IDS[args.camera]
    if camera_id not in calibrations:
        raise SystemExit(f"camera calibration missing for {args.camera}")
    intrinsic, extrinsic = calibrations[camera_id]

    image_path = (
        args.image_root
        / "val"
        / scene_token
        / args.camera
        / f"{frame}.jpg"
    )
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"failed to read image: {image_path}")

    base_xy = np.asarray(base_record["trajectories"][0], dtype=np.float64)[:, :2]
    selected_xy = np.asarray(
        candidate_record["trajectories"][candidate_index],
        dtype=np.float64,
    )[:, :2]
    base_pixels, base_depth = _project_to_image(base_xy, intrinsic, extrinsic)
    candidate_pixels, candidate_depth = _project_to_image(
        selected_xy,
        intrinsic,
        extrinsic,
    )
    _draw_path(
        image,
        base_pixels,
        base_depth,
        color=(255, 120, 0),
    )
    _draw_path(
        image,
        candidate_pixels,
        candidate_depth,
        color=(0, 120, 255),
    )

    nav_command = scene_record.get("meta_info", {}).get("intent")
    if nav_command is None:
        command_index = int(scene_record.get("trajectory", {}).get("nav_command", 0))
        nav_command = ("GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT")[command_index]
    cv2.putText(
        image,
        f"NAV: {nav_command} | calibrated projection | z=0 m",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"NAV: {nav_command} | calibrated projection | z=0 m",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    legend = image.copy()
    cv2.rectangle(legend, (20, 62), (610, 142), (0, 0, 0), -1)
    image[:] = cv2.addWeighted(legend, 0.58, image, 0.42, 0)
    cv2.line(image, (38, 88), (100, 88), (255, 120, 0), 7, cv2.LINE_AA)
    cv2.putText(
        image,
        f"RL base  RFS {base_rfs:.2f}",
        (115, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(image, (38, 123), (100, 123), (0, 120, 255), 7, cv2.LINE_AA)
    cv2.putText(
        image,
        f"Best candidate #{candidate_index}  RFS {candidate_rfs[candidate_index]:.2f}",
        (115, 131),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    manifest = {
        "token": args.token,
        "camera": args.camera,
        "candidate_index": candidate_index,
        "base_rfs": base_rfs,
        "candidate_rfs": float(candidate_rfs[candidate_index]),
        "projection": "Waymo camera calibration; ego trajectory z=0 m",
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
