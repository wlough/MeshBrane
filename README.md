# MeshBrane

![](data/images/time_series_fig.png)

A library for modeling biological membranes.

## Installation

```bash
git clone --recursive git@github.com:wlough/MeshBrane.git
cd MeshBrane
python install.sh
```

See `python install.sh -h` for a complete list of optional flags.

## Dependencies

Installation requires

* [Eigen](https://gitlab.com/libeigen/eigen)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)
* [ffmpeg](https://github.com/FFmpeg/FFmpeg)

## Examples

```bash
build/bin/rigid_spindle_sim data/parameter_files/rigid_spindle_params.yaml
```

<!-- ```bash
build/bin/param_sweep data/parameter_files/phase_diagram_sweep_radius_p65_p45_p25_p15_force_5e1_1e2_2e2_4e2.yaml
``` -->
