# Contributing

Please describe the problem, expected behavior, and reproduction command in an
issue or pull request. Keep changes to numerical methods separate from changes
to packaging and documentation.

## Source checks

Use Python 3.10 or later and install PyYAML. These checks do not require a GPU:

```bash
python -m pip install PyYAML
python -m unittest discover -s tests -v
python -m compileall -q lowlight_underwater scripts
```

CI also builds a source distribution and a wheel and verifies that the CUDA
sources, GLM headers, configuration examples, and licenses are included.

For model or CUDA changes, report the runtime versions, GPU, scene, training
command, and numerical or rendering checks. CPU checks alone do not validate
CUDA gradients or reconstruction quality.

## Repository scope

Keep this repository focused on the model, rasterizer, supported scene configs,
and user documentation. Keep datasets, checkpoints, internal research notes,
parameter sweeps, review correspondence, and generated results outside it.
Document the input requirements when adding a supported scene configuration.
Preserve third-party copyright and license notices.

See [release instructions](docs/RELEASING.md) for packaging checks.
