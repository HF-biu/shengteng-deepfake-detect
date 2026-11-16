"""Image/video CLI and shared application entry point for Ascend OM inference."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from ais_infer import Inference
from utils.ascend_runtime import ROOT, StaticOMSession


def load_config(path=None, device=None):
    with open(path or ROOT / 'config/inference.yaml', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if device is not None:
        config.setdefault('ascend', {})['device_id'] = device
    return config


def infer_frames(engine, frames, input_type='image', frame_offset=0):
    files = [SimpleNamespace(type=input_type, id=i if input_type == 'image' else 0,
                             frame_id=frame_offset + i, content=frame)
             for i, frame in enumerate(frames)]
    detections, segments = engine.infer(files, input_type=input_type)
    states = [state for item in detections.values() for state in item.state_list]
    return [dict(frame_id=int(state[2]), state=state[0],
                 face_detected=bool(np.any(state[3])),
                 deepfake_probability=float(state[1]),
                 manipulation_probability=float(seg.binary),
                 manipulation_mask_ratio=float(seg.cls),
                 score=max(float(state[1]), float(seg.cls)))
            for state, seg in zip(states, segments)]


def infer_path(engine, path, input_type='image'):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if input_type == 'image':
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode image: {}'.format(path))
        return infer_frames(engine, [image])
    if input_type != 'video':
        raise ValueError('input_type must be image or video')
    capture = cv2.VideoCapture(str(path))
    results, frames = [], []
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open video: {}'.format(path))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) == engine.retina_batch:
                results.extend(infer_frames(engine, frames, 'video', len(results)))
                frames = []
        if frames:
            results.extend(infer_frames(engine, frames, 'video', len(results)))
        if not results:
            raise ValueError('Video contains no decodable frames')
    finally:
        capture.release()
    return results


def smoke_test(config):
    report = {}
    for name in ('retina', 'pscc', 'deepfake'):
        session = StaticOMSession(config, name)
        try:
            outputs = session.infer_batch(np.zeros((1,) + session.shape[1:], dtype=np.float32))
            if not outputs or not all(np.isfinite(output).all() for output in outputs):
                raise RuntimeError('{} returned empty or non-finite outputs'.format(name))
            report[name] = dict(input_shape=session.shape,
                                output_shapes=[list(output.shape) for output in outputs])
        finally:
            session.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--device-id', type=int)
    parser.add_argument('--input')
    parser.add_argument('--input-type', choices=('image', 'video'), default='image')
    parser.add_argument('--output', help='Optional JSON result file')
    parser.add_argument('--smoke-test', action='store_true', help='Execute all three OMs on the NPU')
    args = parser.parse_args()
    if not args.smoke_test and not args.input:
        parser.error('--input is required unless --smoke-test is set')
    config = load_config(args.config, args.device_id)
    if args.smoke_test:
        result = smoke_test(config)
    else:
        with Inference(config) as engine:
            result = infer_path(engine, args.input, args.input_type)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
