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
    void printCPU();

#ifdef OPENCL
    void launchGPU();
    void printGPU();
#endif

private:
    size_t localDim[3];
    size_t loops;
    size_t cpuHaloWidth[6];
    size_t mpiNum[3];

#ifdef OPENCL
    size_t gpuDim[3];
    size_t gpuHaloWidth[6];
    size_t cpuForGpuHaloWidth[6];
    bool hybrid;
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

    size_t left, right, top, bottom, front, back;

#ifdef OPENCL
    double **gpudat;
    cl::Buffer *cl_gpudat;
    cl::Buffer cl_valuesPerPointPerBuffer;
#endif

};

#endif // SAMPLE_H
