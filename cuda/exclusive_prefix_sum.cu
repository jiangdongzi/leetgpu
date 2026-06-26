#include <cuda_runtime.h>

/*
Input values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
Input flags:  [  1,   0,   0,   1,   0,   1]

Segments:     [1.0, 2.0, 3.0] | [4.0, 5.0] | [6.0]

Output:       [0.0, 1.0, 3.0,   0.0, 4.0,   0.0]
*/
// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {

}