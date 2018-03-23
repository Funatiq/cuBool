#ifndef CONFIG_H
#define CONFIG_H

// #define BLOCKSPERGRID 1024
#define TILE_WIDTH 32

#ifndef THREADSPERBLOCK
#define THREADSPERBLOCK 128
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

// #ifndef UINT32_MAX
// #define UINT32_MAX  ((uint32_t)-1)
// #endif

#endif