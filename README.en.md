# Pano3DBench

*[Русская версия — README.md](README.md) (main).*

A single, **reproducible** framework for running and comparing holistic indoor-scene
understanding methods from one 360° panorama. It compares three methods —
**DeepPanoContext (DPC)**, **Im3D-Pano** and **Total3D-Pano** — under one evaluation
protocol on two sets: **iGibson** and our own **3D-FRONT** panorama set.

Key idea: the three methods are **configurations of one DeepPanoContext code base**, not
three separate repositories. They share the detector, the layout estimator (HorizonNet)
and the oriented-box estimator (BEN), and differ in the shape head (LDIF or MGN) and in the
relation optimization on the scene graph network (Scene-GCN). So the toolbox runs them
along one path and maps every result to one representation, `<layout, objects, relations>`.

## Getting started

```bash
make setup    # creates .venv and installs the package with test dependencies
make test     # tests, must end with "passed"
make smoke    # demo run without a GPU or models
```

Without `make` (standard library only):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m pano3dbench run --config configs/demo/smoke.json
```

## Reproducing the paper

| Set      | Document                           | What is reproduced                                                                      |
|----------|------------------------------------|-----------------------------------------------------------------------------------------|
| iGibson  | [docs/igibson.md](docs/igibson.md) | five method configurations, two 11-class control rows, the LDIF vs MGN shape comparison |
| 3D-FRONT | [docs/front3d.md](docs/front3d.md) | training the components on our set and evaluating on its test part                      |

The documents in `docs/` are in Russian.

A paper row can be produced along two routes that give the same numbers:

```bash
pano3dbench run --config configs/igibson/runs/dpc.yaml          # through the model environment
pano3dbench run --config configs/igibson/runs/dpc_rescore.yaml  # from converted predictions, no GPU
```

**Weights trained by us** (shape heads on iGibson):
[release `weights-v1`](https://github.com/HoloPano-dissertation/pano3dbench/releases/tag/weights-v1) —
`total3d-pano_mgn_igibson.pth` (MGN for Total3D-Pano), `im3d-pano_ldif_igibson.pth` (LDIF
for the trained Im3D-Pano), `SHA256SUMS.txt`, `MANIFEST.md`. The authors' DPC weights are
public in the DeepPanoContext release and are not redistributed here.

## What the toolbox does

`pano3dbench run --config <cfg>` runs

```text
config -> dataset (manifest + ground truth) -> model -> metrics -> report
```

and writes to `artifacts/<name>/` the predictions, per-scene values, a summary (mean,
median, 95% confidence interval), a report and `run.json` with the config, seed, git
revision and environment info.

The model is **never imported** into the toolbox process: it runs in its own environment
and exchanges JSON files, so methods may need incompatible Python, PyTorch and CUDA
versions. A new method plugs in through a runner and an environment spec, with no change
to the core.

The comparison protocol is **oriented 3D-IoU of objects at threshold 0.15**, plus layout
IoU, object collisions, objects outside the layout and penetration. Shape, the only thing
that separates LDIF from MGN, is measured by `mesh_chamfer` and `mesh_fscore`.

## Layout

```text
pano3dbench/
├── configs/
│   ├── environments/        model environments (dpc, echo)
│   ├── protocols/           shared metric protocol for any set
│   ├── demo/                demo runs without a GPU or data; scene/ is the test scene
│   ├── igibson/
│   │   ├── protocols/       protocol of the 11-class control row
│   │   ├── runs/            <method>.yaml through the environment, <method>_rescore.yaml from converted predictions
│   │   └── dpc/             DPC's own configs for iGibson
│   └── front3d/
│       ├── front3d_to_igibson.json  mapping of 3D-FRONT classes to the iGibson vocabulary
│       ├── dpc/             DPC configs on 3D-FRONT: component training and inference
│       └── runs/            paper rows from saved predictions
├── docs/                    documentation (Russian); configs/README.md and scripts/README.md list every config and script,
│                            src/pano3dbench/README.md and the README of each core folder list every source file
├── runners/                 model runners for isolated environments
├── scripts/
│   ├── dpc/                 DPC prediction conversion, DPC code patches
│   ├── front3d/             method formats, evaluation view lists and checks of the 3D-FRONT set
│   └── igibson/             iGibson ground truth and MGN targets
├── src/pano3dbench/         the core
├── tests/                   tests: core, runners, environments, front3d
├── training/                detector and HorizonNet training on 3D-FRONT
└── pyproject.toml
```

The 3D-FRONT panorama set is built by a separate generator,
[front3d-pano](https://github.com/HoloPano-dissertation/front3d-pano).

## License

MIT, see [LICENSE](LICENSE).
