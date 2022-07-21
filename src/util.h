#ifndef UTIL_H
#define UTIL_H

void check(cudaError_t result, char const *const func, const char *const file,
           int const line);

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line);

/**
 * @brief Check if a CUDA function returns with an error
 */
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

/**
 * @brief get the last error raised by CUDA
 * 
 */
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

/**
 * @brief compute the time interval between two different instants, in ms
 * 
 * @param start: start of the interval 
 * @param end: end of the interval
 * @return int: the interval, in ms
 */

int compute_interval_ms(struct timespec *start, struct timespec *end);

#endif