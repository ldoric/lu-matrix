# LU Matrix Decomposition

## How to Run
```bash
g++ -fopenmp -O2 lu_matrix.cpp -o lu_matrix.exe
./lu_matrix.exe
```

# Result Example
```bash
C:\Users\doric\Desktop\lu_matrix>lu_matrix.exe
Starting LU Decomposition with matrix size: 512
Sequential LU Decomposition | Execution Time: 256 ms
Blocked LU Factorization completed in 70.9999 ms

=== Last few elements of matrices ===
Lower matrix L (last 3x3 block):
1       -5.32952e-019   2.24202e-018
0.0393746       1       -2.79109e-018
0.0393746       0.0449657       1

Upper matrix U (last 3x3 block):
1.76968 0.0833216       0.0841114
2.81893e-018    1.78004 0.0807995
2.6834e-018     -1.0842e-019    1.77717
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