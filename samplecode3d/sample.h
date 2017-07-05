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
    size_t localDim[3];
    size_t gpuDim[3];
    size_t loops;
    size_t cpuHaloWidth[6];
    size_t gpuHaloWidth[6];
    size_t cpuForGpuHaloWidth[6];
    size_t mpiNum[3];
    bool hybrid;

    Tausch3D<double> *tausch;

    TauschHaloSpec *localHaloSpecs;
    TauschHaloSpec *remoteHaloSpecs;
    TauschHaloSpec *localHaloSpecsGpu;
    TauschHaloSpec *remoteHaloSpecsGpu;
    TauschHaloSpec *localHaloSpecsCpuForGpu;
    TauschHaloSpec *remoteHaloSpecsCpuForGpu;

    double **dat;
    size_t numBuffers;
    size_t *valuesPerPointPerBuffer;

    size_t left, right, top, bottom, front, back;

    double **gpudat;
    cl::Buffer *cl_gpudat;
    cl::Buffer cl_valuesPerPointPerBuffer;

};

#endif // SAMPLE_H
