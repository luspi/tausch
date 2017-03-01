#ifndef PARAMSTRUCT
#define PARAMSTRUCT

#include "ctausch.h"

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
    CTausch *tausch;
    cl_mem clGpu;
    int printMpiRank;
};

#endif
