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
    explicit Sample(int localDimX, int localDimY, int localDimZ, int gpuDimX, int gpuDimY, int gpuDimZ, int loops, int haloWidth, int mpiNumX = 0, int mpiNumY = 0, int mpiNumZ = 0, bool cpuonly = false, int clWorkGroupSize = 64, bool giveOpenCLDeviceName = false);
    ~Sample();

    void printCPU();
    void printGPU();

    void launchCPU();
    void launchGPU();
private:
    int dimX, dimY, dimZ, gpuDimX, gpuDimY, gpuDimZ;
    int mpiRank, mpiSize;

    real_t *datCPU;
    real_t *datGPU;
    cl::Buffer cl_datGpu;
    int loops;
    Tausch3D *tausch;
    bool cpuonly;
    int haloWidth;

};

#endif
