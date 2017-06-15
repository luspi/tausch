#ifndef SAMPLE_H
#define SAMPLE_H

#include <mpi.h>
#include <tausch/tausch.h>
#include <iomanip>

class Sample {

public:
    explicit Sample(size_t *localDim, size_t *gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, size_t *mpiNum, bool buildlog, bool hybrid);
    ~Sample();

    void launchCPU();
    void launchGPU();

    void printCPU();
    void printGPU();

private:
    bool hybrid;

    size_t localDim[2];
    size_t gpuDim[2];
    size_t loops;
    size_t cpuHaloWidth[4];
    size_t gpuHaloWidth[4];
    size_t cpuForGpuHaloWidth[4];
    size_t mpiNum[2];

    Tausch2D<double> *tausch;

    TauschHaloSpec *localHaloSpecsCpu;
    TauschHaloSpec *remoteHaloSpecsCpu;
    TauschHaloSpec *localHaloSpecsGpu;
    TauschHaloSpec *remoteHaloSpecsGpu;
    TauschHaloSpec *localHaloSpecsCpuForGpu;
    TauschHaloSpec *remoteHaloSpecsCpuForGpu;
    double *dat1, *dat2;
    size_t numBuffers;
    size_t valuesPerPointPerBuffer[2];

    size_t left, right, top, bottom;

    double *gpudat1, *gpudat2;
    cl::Buffer cl_gpudat1, cl_gpudat2;
    cl::Buffer cl_valuesPerPointPerBuffer;

};

#endif // SAMPLE_H
