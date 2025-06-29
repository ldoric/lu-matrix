#include <iostream>
#include <cmath>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

const int MATRIX_SIZE = 512;     // Matrix dimension
const int BLOCK_SIZE = 64;       // Block dimension for optimization

double matrix_A[MATRIX_SIZE][MATRIX_SIZE], lower_L[MATRIX_SIZE][MATRIX_SIZE], upper_U[MATRIX_SIZE][MATRIX_SIZE];

void setup_initial_matrix() {
    // Initialize matrix A with test values (ensuring non-singular)
    for (int row = 0; row < MATRIX_SIZE; row++)
        for (int col = 0; col < MATRIX_SIZE; col++)
            matrix_A[row][col] = (row == col) ? 2.5 : 0.8;

    // Initialize L and U matrices to zero
    for (int row = 0; row < MATRIX_SIZE; row++)
        for (int col = 0; col < MATRIX_SIZE; col++)
            lower_L[row][col] = upper_U[row][col] = 0.0;
}

void sequential_lu_decomposition() {
    auto time_start = high_resolution_clock::now();

    for (int pivot = 0; pivot < MATRIX_SIZE; pivot++) {
        // Compute upper triangular matrix elements
        for (int col = pivot; col < MATRIX_SIZE; col++) {
            double accumulator = 0;
            for (int idx = 0; idx < pivot; idx++)
                accumulator += (lower_L[pivot][idx] * upper_U[idx][col]);
            upper_U[pivot][col] = matrix_A[pivot][col] - accumulator;
        }

        // Compute lower triangular matrix elements
        for (int row = pivot; row < MATRIX_SIZE; row++) {
            if (pivot == row)
                lower_L[pivot][pivot] = 1.0;  // Diagonal elements of L are 1
            else {
                double accumulator = 0;
                for (int idx = 0; idx < pivot; idx++)
                    accumulator += (lower_L[row][idx] * upper_U[idx][pivot]);
                lower_L[row][pivot] = (matrix_A[row][pivot] - accumulator) / upper_U[pivot][pivot];
            }
        }
    }

    auto time_end = high_resolution_clock::now();
    auto elapsed_time = duration_cast<milliseconds>(time_end - time_start);
    cout << "Sequential LU Decomposition | Execution Time: " << elapsed_time.count() << " ms" << endl;
}

void perform_blocked_lu_factorization() {
    for (int block_start = 0; block_start < MATRIX_SIZE; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, MATRIX_SIZE);

        // Phase 1: Factorize the diagonal block
        for (int row = block_start; row < block_end; ++row) {
            for (int col = block_start; col < block_end; ++col) {
                double temp_sum = 0;
                for (int k = block_start; k < row; ++k)
                    temp_sum += lower_L[row][k] * upper_U[k][col];
                upper_U[row][col] = matrix_A[row][col] - temp_sum;
            }

            for (int col = block_start; col < block_end; ++col) {
                if (row == col)
                    lower_L[row][row] = 1.0;
                else {
                    double temp_sum = 0;
                    for (int k = block_start; k < row; ++k)
                        temp_sum += lower_L[col][k] * upper_U[k][row];
                    lower_L[col][row] = (matrix_A[col][row] - temp_sum) / upper_U[row][row];
                }
            }
        }

        // Phase 2: Process U blocks on the right side of diagonal
        #pragma omp parallel for collapse(2)
        for (int row = block_start; row < block_end; ++row) {
            for (int col = block_end; col < MATRIX_SIZE; ++col) {
                double temp_sum = 0;
                for (int k = block_start; k < row; ++k)
                    temp_sum += lower_L[row][k] * upper_U[k][col];
                upper_U[row][col] = matrix_A[row][col] - temp_sum;
            }
        }

        // Phase 3: Process L blocks below diagonal
        #pragma omp parallel for collapse(2)
        for (int row = block_end; row < MATRIX_SIZE; ++row) {
            for (int col = block_start; col < block_end; ++col) {
                double temp_sum = 0;
                for (int k = block_start; k < col; ++k)
                    temp_sum += lower_L[row][k] * upper_U[k][col];
                lower_L[row][col] = (matrix_A[row][col] - temp_sum) / upper_U[col][col];
            }
        }

        // Phase 4: Update the remaining submatrix
        #pragma omp parallel for collapse(2)
        for (int row = block_end; row < MATRIX_SIZE; ++row) {
            for (int col = block_end; col < MATRIX_SIZE; ++col) {
                double temp_sum = 0;
                for (int k = block_start; k < block_end; ++k)
                    temp_sum += lower_L[row][k] * upper_U[k][col];
                matrix_A[row][col] -= temp_sum;
            }
        }
    }
}

void display_final_matrix_elements() {
    cout << "\n=== Last few elements of matrices ===" << endl;
    
    // Show last 3x3 block of L matrix
    cout << "Lower matrix L (last 3x3 block):" << endl;
    for (int i = MATRIX_SIZE - 3; i < MATRIX_SIZE; i++) {
        for (int j = MATRIX_SIZE - 3; j < MATRIX_SIZE; j++) {
            cout << lower_L[i][j] << "\t";
        }
        cout << endl;
    }
    
    // Show last 3x3 block of U matrix
    cout << "\nUpper matrix U (last 3x3 block):" << endl;
    for (int i = MATRIX_SIZE - 3; i < MATRIX_SIZE; i++) {
        for (int j = MATRIX_SIZE - 3; j < MATRIX_SIZE; j++) {
            cout << upper_U[i][j] << "\t";
        }
        cout << endl;
    }
}

int main() {
    cout << "Starting LU Decomposition with matrix size: " << MATRIX_SIZE << endl;
    
    setup_initial_matrix();
    sequential_lu_decomposition();

    setup_initial_matrix();  // Reset matrices for blocked version
    double start_time = omp_get_wtime();
    perform_blocked_lu_factorization();
    double end_time = omp_get_wtime();

    cout << "Blocked LU Factorization completed in " << (end_time - start_time) * 1000 << " ms" << endl;
    
    display_final_matrix_elements();

    return 0;
}
