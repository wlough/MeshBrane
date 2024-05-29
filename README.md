# MeshBrane

Create environment:

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
cd tests
g++ -o test.exe \
-I /usr/include/eigen3 \
-I /usr/local/include \
-I ../include \
ply_utils_test.cpp \
../source/ply_utils.cpp
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
