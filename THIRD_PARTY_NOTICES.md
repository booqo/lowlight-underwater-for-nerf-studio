# Third-party notices

The root MIT license applies to original project contributions. It does not
replace the licenses or notices of incorporated third-party code.

| Component | Attribution / source | License retained |
| --- | --- | --- |
| Nerfstudio-derived training, evaluation, rendering, and model code | Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors; see existing headers in `lowlight_underwater/{train,eval,render,lowlight_underwater_model}.py` | [Apache-2.0](licenses/Apache-2.0.txt) |
| CUDA rasterizer and Gaussian utility routines | Derived from [gsplat](https://github.com/nerfstudio-project/gsplat), Copyright 2023 The Nerfstudio Team, as recorded in the root license | [Apache-2.0](licenses/Apache-2.0.txt) |
| Bundled GLM headers | G-Truc Creation; see bundled copyright notice | [GLM copying.txt](lowlight_underwater/cudalight/csrc/third_party/glm/copying.txt) |

The rasterizer and model have been modified for underwater image formation and
low-light enhancement. Existing source-level attribution is retained. The
spherical-harmonic implementation in `cudalight/_torch_impl.py` also records its
svox2 origin; preserve that attribution when redistributing it.

External dependencies and datasets are distributed separately and retain their
own terms. This inventory records the notices available in this repository;
maintainers should confirm the provenance of contributed code and assets before
publishing a release.
