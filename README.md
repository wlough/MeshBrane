# MeshBrane


## To do
- decimation/subdivision
- cotan laplacian
- laplacian-->gradient



## Create environment:

```
python3.10 -m venv ~/.env/meshbrane
```

Install python requirements:

```
pip install -r requirements.txt
```

Save python requirements:

```
pip freeze > requirements.txt
```

## tests for ply_utils
```bash
cd tests/cpp
g++ -o test.exe \
-I /usr/include/eigen3 \
-I /usr/local/include \
-I ../../include \
ply_utils_test.cpp \
../../src/cpp/ply_utils.cpp
```


## tests for pretty_pictures
```bash
cd tests
g++ -o test.exe \
-I /usr/include/eigen3 \
-I /usr/local/include \
-I ../include \
pretty_pictures_test.cpp \
../source/ply_utils.cpp
```

## tests for half_edge_base_utils
```bash
g++ -o half_edge_base_utils_test.exe \
-I /usr/include/eigen3 \
-I /usr/local/include \
half_edge_base_utils_test.cpp \
half_edge_base_utils.cpp 
```


### Directory structure

MeshBrane
├── bin
├── build
├── data
│   ├── config
│   └── ply
│       ├── ascii
│       └── binary
├── docs
├── include
├── lib
├── output
├── prototyping
├── scripts
├── src
│   ├── cpp
│   └── python
├── tests
│   ├── cpp
│   └── python
└── README.md
