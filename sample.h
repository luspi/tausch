#ifndef SAMPLE_H
#define SAMPLE_H

#include "tausch.h"
#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>

class Sample {

public:
    explicit Sample(int localDimX, int localDimY, double portionGPU, int loops, int mpiNumX = 0, int mpiNumY = 0, bool cpuonly = false);

private:
    int dimX, dimY, gpuDimX, gpuDimY;
    int mpiRank, mpiSize;

    void EveryoneOutput(const std::string &inMessage);

    void printCPU(double *dat);
    void printGPU(double *dat);

};

#endif
