#ifndef SAMPLE_H
#define SAMPLE_H

#include "tausch.h"
#include <fstream>
#include <cmath>
#include <iomanip>

class Sample {

public:
    explicit Sample();

private:
    int dimX, dimY, gpuDimX, gpuDimY;
    int mpiRank, mpiSize;

    void EveryoneOutput(const std::string &inMessage);

    void printCPU(double *dat);
    void printGPU(double *dat);

};

#endif
