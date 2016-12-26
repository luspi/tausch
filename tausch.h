#ifndef _TAUSCH_H
#define _TAUSCH_H

#include <vector>
#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sstream>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class Tausch {

public:
    explicit Tausch(int dim_x, int dim_y);
    ~Tausch();

    void startTausch();
    void completeTausch();
    void startAndCompleteTausch() { startTausch(); completeTausch(); }

    cl::Platform cl_platform;
    cl::Device cl_default_device;
    cl::Context cl_context;
    cl::CommandQueue cl_queue;
    cl::Program cl_programs;

    void setCPUHaloInfo(int halowidth, double *dat) { width_ext = halowidth; cpudat = dat; }
    void setGPUHaloInfo(int halowidth, cl::Buffer &dat, int *coords) { width_int = halowidth; gpudat = dat; this->coords = coords; }

private:
    int dim_x, dim_y;
    double *cpudat;
    int *coords;
    int width_ext, width_int;
    cl::Buffer gpudat;

    int mpi_rank, mpi_size;
    int mpi_dim_x, mpi_dim_y;

    void setupOpenCL();

    double *collectCPUBoundaryData();
    void distributeCPUHaloData();

    double *recvbuffer;

    int sendRecvCount;
    MPI_Request *sendRecvRequest;

};

#endif
