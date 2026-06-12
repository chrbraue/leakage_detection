# Voronoi-Based Vacuum Leakage Detection in Composite Manufacturing

This directory contains the data and source code required to reproduce the results presented in the paper *Voronoi-Based Vacuum Leakage Detection in Composite Manufacturing* by Christoph Brauer, Arne Hindersmann, and Timo de Wolff.

## Requirements

The code was developed and tested with Python 3.7 and the package versions listed in `requirements.txt`.

Install the required packages using:

```bash
pip install -r requirements.txt
```

Python 3.7 is no longer supported, so this is more of a reproducibility snapshot than a modern environment recommendation.

## Quick Start

To reproduce the results presented in the paper:

1. Generate the cleaned and filtered datasets:

```bash
python prepare_data.py
```

2. Reproduce the illustrative Voronoi example:

```bash
python random_example.py
```

3. Reproduce the experimental leakage localization results:

```bash
python wing_example.py
```

Generated figures will be stored in the `figures` directory.

## Data

The `data` subfolder contains the following files:

* `data.csv`: A dataset comprising 853 experiments involving either one or two leakage locations. Each sample contains the leak coordinates and the corresponding flow rates measured at ten vacuum connections.
* `contour.csv`: Two-dimensional coordinates describing the mold contour.
* `vacuumports.csv`: Two-dimensional coordinates specifying the positions of the ten vacuum connections on the mold surface.

## Figures

The `figures` subfolder initially contains only the placeholder file `dummy.txt` to ensure that the directory is tracked by Git. After executing `random_example.py` and `wing_example.py`, all generated figures will be stored in this folder.

## Python Code

The repository includes the following Python scripts:

* `leak_localization.py` defines the classes `ClassicVoronoi` and `RefinedVoronoi`, which form the core of the proposed leakage detection methodology.
* `random_example.py` reproduces the illustrative example presented in the paper, including the classical and refined Voronoi diagrams as well as the corresponding Delaunay triangulation. This script provides a convenient introduction to the functionality of the implemented classes.
* `prepare_data.py` generates the cleaned and filtered datasets required for the empirical analysis.
* `wing_example.py` applies the Voronoi-based leakage detection approach to the experimental dataset and reproduces the figures and numerical results reported in the paper.

## Contact

For questions regarding the code, data, or methodology, please contact:

Christoph Brauer
`christoph.brauer@dlr.de`

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{brauer2026voronoi,
  title={Voronoi-Based Vacuum Leakage Detection in Composite Manufacturing},
  author={Brauer, Christoph and Hindersmann, Arne and de Wolff, Timo},
  journal={arXiv preprint arXiv:2603.29980},
  year={2026}
}
```

