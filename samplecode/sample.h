#ifndef SAMPLE_H
#define SAMPLE_H

#include <tausch.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>

typedef double real_t;

class Sample {

public:
    explicit Sample(int localDimX, int localDimY, real_t portionGPU, int loops, int mpiNumX = 0, int mpiNumY = 0, bool cpuonly = false, int clWorkGroupSize = 64, bool giveOpenCLDeviceName = false);
    ~Sample();

    void printCPU();
    void printGPU();

    void launchCPU();
    void launchGPU();
private:
    int dimX, dimY, gpuDimX, gpuDimY;
    int mpiRank, mpiSize;

    real_t *datCPU;
    real_t *datGPU;
    cl::Buffer cl_datGpu;
    int loops;
    Tausch *tausch;
    bool cpuonly;

};

#endif
