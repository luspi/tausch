#ifndef _TAUSCH_H
#define _TAUSCH_H

#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iostream>
#include <thread>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class Tausch {

    enum {
        Left = 0,
        Right,
        Top,
        Bottom
    };
    typedef int Edge;

public:
    explicit Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool withOpenCL = false, bool setupOpenCL = false, int clLocalWorkgroupSize = 64, bool giveOpenCLDeviceName = false);
    ~Tausch();

    void setOpenCLInfo(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue);

    void postCpuReceives();

    void performCpuToCpuTausch() { startCpuTauschLeft(); startCpuTauschRight(); completeCpuTauschLeft(); completeCpuTauschRight();
                              startCpuTauschTop(); startCpuTauschBottom(); completeCpuTauschTop(); completeCpuTauschBottom(); }

    void performCpuToGpuTausch() { startCpuToGpuTausch(); completeCpuToGpuTausch(); }

    void performCpuToCpuAndCpuToGpuTausch() { startCpuTauschLeft(); startCpuTauschRight(); startCpuToGpuTausch();
                                         completeCpuTauschLeft(); completeCpuTauschRight(); startCpuTauschTop(); startCpuTauschBottom();
                                         completeCpuToGpuTausch(); completeCpuTauschTop(); completeCpuTauschBottom(); }

    void performGpuToCpuTausch() { startGpuToCpuTausch(); completeGpuToCpuTausch(); }

    void startCpuToGpuTausch();
    void startGpuToCpuTausch();

    void completeCpuToGpuTausch();
    void completeGpuToCpuTausch();

    void startCpuTauschLeft();
    void startCpuTauschRight();
    void startCpuTauschTop();
    void startCpuTauschBottom();

    void completeCpuTauschLeft();
    void completeCpuTauschRight();
    void completeCpuTauschTop();
    void completeCpuTauschBottom();

    void syncCpuAndGpu(bool iAmTheCPU);


    void setHaloWidth(int haloWidth) { this->haloWidth = 1; }
    void setCPUData(double *dat);
    void setGPUData(cl::Buffer &dat, int gpuWidth, int gpuHeight);
    bool isGpuEnabled() { return gpuEnabled; }

private:
    void EveryoneOutput(const std::string &inMessage);

    cl::Context cl_context;
    cl::CommandQueue cl_queue;

    int localDimX, localDimY;
    double *cpuData;
    cl::Buffer gpuData;

    int haloWidth;
    int gpuWidth, gpuHeight;
    cl::Buffer cl_gpuWidth, cl_gpuHeight;

    int mpiRank, mpiSize;
    int mpiNumX, mpiNumY;

    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();

    double *cpuToCpuSendBuffer;
    double *cpuToCpuRecvBuffer;

    double *cpuToGpuBuffer;
    double *gpuToCpuBuffer;

    cl::Platform cl_platform;
    cl::Device cl_defaultDevice;
    cl::Program cl_programs;

    cl::Buffer cl_gpuToCpuBuffer;

    bool gpuEnabled;
    bool gpuInfoGiven;
    bool cpuInfoGiven;
    bool cpuRecvsPosted;

    bool cpuLeftStarted;
    bool cpuRightStarted;
    bool cpuTopStarted;
    bool cpuBottomStarted;

    bool cpuToGpuStarted;
    bool gpuToCpuStarted;

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

    int cl_kernelLocalSize;



    MPI_Request cpuToCpuLeftSendRequest;
    MPI_Request cpuToCpuRightSendRequest;
    MPI_Request cpuToCpuTopSendRequest;
    MPI_Request cpuToCpuBottomSendRequest;
    MPI_Request cpuToCpuLeftRecvRequest;
    MPI_Request cpuToCpuRightRecvRequest;
    MPI_Request cpuToCpuTopRecvRequest;
    MPI_Request cpuToCpuBottomRecvRequest;

    int syncpointCpu;
    int syncpointGpu;

};

#endif
