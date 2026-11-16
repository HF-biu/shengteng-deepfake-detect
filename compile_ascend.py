"""Compile the supplied ONNX models for a specific Ascend SoC using ATC."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from ascend_infer import load_config
from utils.ascend_runtime import ROOT, SAMPLE_SHAPES

ONNX_MODELS = {'retina': 'FaceDetector.onnx', 'pscc': 'pscc_net.onnx',
               'deepfake': 'cfl_128.onnx'}


def make_command(atc, path, destination, soc, input_name, shape):
    return [atc, '--framework=5', '--model=' + str(path),
            '--output=' + str(destination), '--soc_version=' + soc,
            '--input_format=NCHW',
            '--input_shape={}:{}'.format(input_name, ','.join(map(str, shape))),
            '--output_type=FP32', '--log=info']


def inspect_input(path, shape):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    constants = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in constants]
    if len(inputs) != 1:
        raise ValueError('{} must have exactly one input'.format(path))
    tensor = inputs[0].type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT or len(tensor.shape.dim) != 4:
        raise ValueError('{} must have one float32 NCHW input'.format(path))
    for dim, required in zip(tensor.shape.dim, shape):
        if dim.HasField('dim_value') and dim.dim_value > 0 and dim.dim_value != required:
            raise ValueError('{} has fixed dimension {}, requested {}; re-export ONNX '
                             'or use its original batch size'.format(path, dim.dim_value, required))
    return inputs[0].name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--soc-version', default='Ascend310P3')
    parser.add_argument('--output-dir', default=str(ROOT / 'build/ascend310p3'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    directory = Path(args.output_dir).resolve()
    atc = shutil.which('atc')
    if not atc and not args.dry_run:
        parser.error('atc not found; source the CANN toolkit set_env.sh first')
    plans = []
    for name, filename in ONNX_MODELS.items():
        source = ROOT / filename
        if not source.is_file():
            raise FileNotFoundError(str(source))
        batch = config['batch_size'][name]
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError('Invalid batch_size.' + name)
        shape = (batch,) + SAMPLE_SHAPES[name]
        input_name = inspect_input(source, shape)
        destination = directory / source.stem
        if destination.with_suffix('.om').exists():
            raise FileExistsError('Use a new output directory: ' + str(destination))
        command = make_command(atc or 'atc', source, destination, args.soc_version, input_name, shape)
        plans.append((name, destination, command))
    for name, destination, command in plans:
        print(' '.join(shlex.quote(part) for part in command))
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
            if not destination.with_suffix('.om').is_file():
                raise RuntimeError('ATC did not produce ' + str(destination.with_suffix('.om')))
    if not args.dry_run:
        import yaml
        config.setdefault('ascend', {})['models'] = {
            name: str(destination.with_suffix('.om')) for name, destination, _ in plans}
        (directory / 'inference.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        (directory / 'build_manifest.json').write_text(json.dumps({
            'soc_version': args.soc_version,
            'commands': [command for _, _, command in plans]}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
