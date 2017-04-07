#ifndef PARAMSTRUCT
#define PARAMSTRUCT

#include <tausch/tausch.h>

struct param_struct {
    int localDimX, localDimY;
    double portionGPU;
    int mpiNumX;
    int mpiNumY;
    int loops;
    bool cpuonly;
    int workgroupsize;
    bool giveOpenClDeviceName;
    double *cpu;
    double *gpu;
    int gpuDimX;
    int gpuDimY;
    CTausch2D *tausch;
    cl_mem clGpu;
    int printMpiRank;
    int haloWidth;
};

#endif
