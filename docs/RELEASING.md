# Release instructions

Publish releases to
[`booqo/lowlight-underwater-for-nerf-studio`](https://github.com/booqo/lowlight-underwater-for-nerf-studio/releases),
the repository linked in the LLSU article. Use a reviewed commit from this
repository's `main` branch as the release target.

Run these commands from a clean checkout:

```bash
python -m pip install build PyYAML
python -m unittest discover -s tests -v
python -m build
python scripts/check_wheel.py dist/*.whl
```

Check that the source distribution and wheel contain the CUDA sources, bundled
GLM headers, configuration examples, and license notices. Source checks and
packaging checks run on CPU; a runtime smoke test requires the environment
described in the README. Validate training, evaluation, and rendering on a GPU
before reporting end-to-end reproducibility for a release.

Full training validation on a second machine is pending for this release.

To distribute the repository source without Git metadata:

```bash
python scripts/export_source.py --output dist/llsu-source.tar.gz
```

The archive contains a `RELEASE_MANIFEST.sha256` file. By default the exporter
includes tracked, non-ignored files from the working tree. Use
`--include-untracked` only after reviewing the new files. Existing output files
are not overwritten. Review the archive contents and dependency licenses before
distribution.

Keep datasets and trained checkpoints as separately documented downloads.
Keep the README BibTeX and `CITATION.cff` consistent with the published article.
Confirm the rights to any newly contributed code, images, or data before including
them in a release.

Release notes for version 0.1.0 are in [releases/v0.1.0.md](releases/v0.1.0.md).
Attach the reviewed source archive and its SHA-256 checksum to the release.
