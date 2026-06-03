# MeshBrane

![Mitosis timelapse](data/images/time_series_fig.png "Mitosis timelapse")

## Installation

```bash
git clone --recursive git@github.com:wlough/MeshBrane.git
cd MeshBrane
./install.sh
```

See `./install.sh -h` for a complete list of optional flags.

## Dependencies

Installation requires

* [Eigen](https://gitlab.com/libeigen/eigen)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)

## Examples

```bash
build/bin/rigid_spindle_sim data/parameter_files/rigid_spindle_params.yaml
```
