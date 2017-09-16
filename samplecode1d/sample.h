#ifndef SAMPLE_H
#define SAMPLE_H

#include <mpi.h>
#include <tausch/tausch.h>
#include <iomanip>

class Sample {

public:
    explicit Sample(size_t localDim, size_t gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, bool hybrid);
    ~Sample();

    void launchCPU();
    void launchGPU();

    void printCPU();
    void printGPU();

private:
    bool hybrid;

    size_t localDim;
    size_t gpuDim;
    size_t loops;
    size_t cpuHaloWidth[2];
    size_t gpuHaloWidth[2];
    size_t cpuForGpuHaloWidth[2];

    Tausch<double> *tausch;

    TauschHaloSpec *localHaloSpecs;
    TauschHaloSpec *remoteHaloSpecs;
    TauschHaloSpec *localHaloSpecsGpu;
    TauschHaloSpec *remoteHaloSpecsGpu;
    TauschHaloSpec *localHaloSpecsCpuForGpu;
    TauschHaloSpec *remoteHaloSpecsCpuForGpu;

    double **dat;
    size_t numBuffers;
    size_t *valuesPerPointPerBuffer;

    size_t left, right, top, bottom;

    double **gpudat;
    cl::Buffer *cl_gpudat;
    cl::Buffer cl_valuesPerPointPerBuffer;

};

#endif // SAMPLE_H
