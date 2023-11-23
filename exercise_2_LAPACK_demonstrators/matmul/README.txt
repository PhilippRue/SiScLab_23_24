# Tests for zgemm performance

## Used toolchain:

```
module load intel
```


## compile Zgemm test app

```
ifort -qmkl test_matmul.f90 -o test.exe
```
