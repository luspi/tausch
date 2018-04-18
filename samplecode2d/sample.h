#ifndef SAMPLE_H
#define SAMPLE_H

#include <mpi.h>
#include <tausch/tausch.h>
#include <iomanip>

class Sample {

public:
    explicit Sample(size_t *localDim, size_t *gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, size_t *mpiNum, bool buildlog, bool hybrid, bool gpuonly);
    ~Sample();

    void launchCPU();
    void printCPU();

#ifdef OPENCL
    void launchGPU();
    void launchGPUonly();
    void printGPU();
#endif

private:
#ifdef OPENCL
    bool hybrid;
    bool gpuonly;
#endif

    size_t localDim[2];
    size_t cpuHaloWidth[4];
    size_t mpiNum[2];
    size_t loops;
#ifdef OPENCL
    size_t gpuDim[2];
    size_t gpuHaloWidth[4];
    size_t cpuForGpuHaloWidth[4];
#endif

    Tausch<double> *tausch;

    TauschHaloSpec *localHaloSpecsCpu;
    TauschHaloSpec *remoteHaloSpecsCpu;
#ifdef OPENCL
    TauschHaloSpec *localHaloSpecsGpu;
    TauschHaloSpec *remoteHaloSpecsGpu;
    TauschHaloSpec *localHaloSpecsCpuForGpu;
    TauschHaloSpec *remoteHaloSpecsCpuForGpu;
    TauschHaloSpec *localHaloSpecsGpuWithGpu;
    TauschHaloSpec *remoteHaloSpecsGpuWithGpu;
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
