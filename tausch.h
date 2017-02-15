#ifndef _TAUSCH_H
#define _TAUSCH_H

#include <vector>
#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iostream>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class Tausch {

public:
    explicit Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool withOpenCL = false);
    ~Tausch();

    void postCpuReceives();
    void postGpuReceives();

    void startCpuTausch();
    void completeCpuTausch();
    void startAndCompleteCpuTausch() { startCpuTausch(); completeCpuTausch(); }
    void startGpuTausch();
    void completeGpuTausch();
    void startAndCompleteGpuTausch() { startGpuTausch(); completeGpuTausch(); }

    cl::Platform cl_platform;
    cl::Device cl_defaultDevice;
    cl::Context cl_context;
    cl::CommandQueue cl_queue;
    cl::Program cl_programs;

    void setHaloWidth(int haloWidth) { this->haloWidth = haloWidth; }
    void setCPUData(double *dat);
    void setGPUData(cl::Buffer &dat, int gpuWidth, int gpuHeight);

private:
    int localDimX, localDimY;
    double *cpuData;
    cl::Buffer gpuData;

    int haloWidth;
    int gpuWidth, gpuHeight;
    cl::Buffer cl_gpuWidth, cl_gpuHeight;

    int mpiRank, mpiSize;
    int mpiNumX, mpiNumY;

    void setupOpenCL();

    double *cpuToCpuSendBuffer;
    double *cpuToCpuRecvBuffer;
    double *cpuToGpuSendBuffer;
    double *cpuToGpuRecvBuffer;
    double *gpuToCpuSendBuffer;
    double *gpuToCpuRecvBuffer;

    cl::Buffer cl_gpuToCpuSendBuffer;
    cl::Buffer cl_gpuToCpuRecvBuffer;

    bool gpuEnabled;
    bool gpuInfoGiven;
    bool cpuInfoGiven;

    // This refers to the overall domain border
    bool haveLeftBorder;
    bool haveRightBorder;
    bool haveTopBorder;
    bool haveBottomBorder;

    // this refers to inter-partition boundaries
    bool haveLeftBoundary;
    bool haveRightBoundary;
    bool haveTopBoundary;
    bool haveBottomBoundary;

    std::vector<MPI_Request> allCpuRequests;
    std::vector<MPI_Request> allGpuRequests;

    int cl_kernelLocalSize;


};

#endif
