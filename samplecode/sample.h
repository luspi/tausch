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
    explicit Sample(int localDimX, int localDimY, int gpuDimX, int gpuDimY, int loops, int cpuHaloWidth[4], int gpuHaloWidth[4], int mpiNumX = 0, int mpiNumY = 0, bool cpuonly = false, int clWorkGroupSize = 64, bool giveOpenCLDeviceName = false);
    ~Sample();

    void printCPU();
    void printGPU();
    void printCPUStencil();
    void printGPUStencil();

    void launchCPU();
    void launchGPU();
private:
    int dimX, dimY, gpuDimX, gpuDimY;
    int mpiRank, mpiSize;

    real_t *datCPU;
    real_t *datGPU;

    int stencilNumPoints;
    real_t *stencil;
    real_t *stencilGPU;

    cl::Buffer cl_datGpu;
    cl::Buffer cl_stencilGPU;
    int loops;
    Tausch2D *tausch;
    bool cpuonly;
    int cpuHaloWidth[4];
    int gpuHaloWidth[4];

};

#endif
