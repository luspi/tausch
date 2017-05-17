#ifndef SAMPLE_H
#define SAMPLE_H

#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>

#define TAUSCH_OPENCL
#include <tausch/tausch.h>

typedef double real_t;

class Sample {

public:
    explicit Sample(int *localDim, int *gpuDim, int loops, int *cpuHaloWidth, int *gpuHaloWidth, int *mpiNum, bool cpuonly, int clWorkGroupSize, bool giveOpenCLDeviceName);
    ~Sample();

    void printCPU();
    void printCpuStencil();
    void printGPU();
    void printGpuStencil ();

    void launchCPU();
    void launchGPU();
private:
    int dimX, dimY, dimZ, gpuDimX, gpuDimY, gpuDimZ;
    int mpiRank, mpiSize;

    real_t *datCPU;
    real_t *datGPU;

    int stencilNumPoints;
    real_t *stencilCPU;
    real_t *stencilGPU;

    cl::Buffer cl_datGpu;
    cl::Buffer cl_stencilGpu;
    int loops;
    Tausch3D *tausch;
    bool cpuonly;
    int cpuHaloWidth[6];
    int gpuHaloWidth[6];

};

#endif
