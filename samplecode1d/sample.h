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
    void printCPU();

#ifdef OPENCL
    void launchGPU();
    void printGPU();
#endif

private:
    bool hybrid;

    size_t localDim;
    size_t loops;
    size_t cpuHaloWidth[2];
#ifdef OPENCL
    size_t gpuDim;
    size_t gpuHaloWidth[2];
    size_t cpuForGpuHaloWidth[2];
#endif

    Tausch<double> *tausch;

    TauschHaloSpec *localHaloSpecs;
    TauschHaloSpec *remoteHaloSpecs;
#ifdef OPENCL
    TauschHaloSpec *localHaloSpecsGpu;
    TauschHaloSpec *remoteHaloSpecsGpu;
    TauschHaloSpec *localHaloSpecsCpuForGpu;
    TauschHaloSpec *remoteHaloSpecsCpuForGpu;
#endif

    double **dat;
    size_t numBuffers;
    size_t *valuesPerPointPerBuffer;

    size_t left, right, top, bottom;

#ifdef OPENCL
    double **gpudat;
    cl::Buffer *cl_gpudat;
    cl::Buffer cl_valuesPerPointPerBuffer;
#endif

};

#endif // SAMPLE_H
