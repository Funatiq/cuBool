#ifndef CONFIG_H
#define CONFIG_H

// #ifndef THREADSPERBLOCK
// #define THREADSPERBLOCK 512
// #endif
#ifndef WARPSPERBLOCK
#define WARPSPERBLOCK 16
#endif
#ifndef DIM_PARAM
#define DIM_PARAM 20
#endif
#ifndef CPUITERATIONS
#define CPUITERATIONS 100000000
#endif
#ifndef INITIALIZATIONMODE
#define INITIALIZATIONMODE 2
#endif

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 32
#endif

// #ifndef UINT32_MAX
// #define UINT32_MAX  ((uint32_t)-1)
// #endif

#endif