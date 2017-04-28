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
    explicit Sample(int localDimX, int localDimY, real_t portionGPU, int loops, int cpuHaloWidth, int gpuHaloWidth, int mpiNumX = 0, int mpiNumY = 0, bool cpuonly = false, int clWorkGroupSize = 64, bool giveOpenCLDeviceName = false);
    ~Sample();

    void printCPU();
    void printGPU();
    void printCPUStencil();

    void launchCPU();
    void launchGPU();
private:
    int dimX, dimY, gpuDimX, gpuDimY;
    int mpiRank, mpiSize;

    real_t *datCPU;
    real_t *datGPU;

    int stencilNumPoints;
    real_t *stencil;

    cl::Buffer cl_datGpu;
    int loops;
    Tausch2D *tausch;
    bool cpuonly;
    int cpuHaloWidth;
    int gpuHaloWidth;

};

#endif
