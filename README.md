# LU Matrix Decomposition

## How to Run
```bash
g++ -fopenmp -O2 lu_matrix.cpp -o lu_matrix.exe
./lu_matrix.exe
```

# Result Example
```bash
C:\Users\doric\Documents\GitHub\lu-matrix>lu_matrix.exe                                  
Starting LU Decomposition with matrix size: 512
Sequential LU Decomposition | Execution Time: 302 ms
Blocked LU Factorization completed in 59 ms

=== Last few elements of matrices ===
Lower matrix L (last 3x3 block):
1       -3.77187e-017   4.02391e-017
0.43775 1       -2.82341e-017
0.43775 0.3139  1

Upper matrix U (last 3x3 block):
3.02356 1.38332 1.96398
-7.79541e-017   2.47777 1.10425
-5.34512e-017   5.31259e-017    2.45763
```

## What is OpenMP?
OpenMP is a parallel programming library that allows the program to use multiple CPU cores simultaneously for faster computation.

## What This Code Does
This program performs LU decomposition on a 512×512 matrix using two methods:
1. **Sequential**: Traditional row-by-row processing
2. **Blocked**: Cache-optimized processing in 64×64 blocks with parallel execution

The program compares the performance of both methods and displays the last few matrix elements after decomposition.

## Output
- Execution times for both sequential and blocked versions
- Last 3×3 block of the resulting L and U matrices 