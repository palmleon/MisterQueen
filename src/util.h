#ifndef UTIL_H
#define UTIL_H

void check(cudaError_t result, char const *const func, const char *const file,
           int const line);

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line);

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

#endif