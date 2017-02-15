#include "sample.h"

/*****************************
 *
 * This code uses the following test setup
 *
 *     |             20              |
 *   -----------------------------------
 *     |   CPU       5               |
 *     |                             |
 *     |       ---------------       |
 *  2  |   5   |    GPU      |   5   |
 *  0  |       |             |       |
 *     |       ---------------       |
 *     |                             |
 *     |             5               |
 *   -----------------------------------
 *     |                             |
 *
 *  with a halo width of 1 (for any halo).
 *
 *******************************/

Sample::Sample() {

    // obtain MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // the overall x and y dimension of the local partition
    int dim_x = 20, dim_y = 20;
    int gpuDimX = dim_x*0.5, gpuDimY = dim_y*0.5;

    Tausch tau(dim_x, dim_y, std::sqrt(mpi_size), std::sqrt(mpi_size), true);

    // the width of the halos
    int halowidth = 1;

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dim_x+2)*(dim_y+2);
    double *dat = new double[num]{};

    tau.setHaloWidth(1);

    tau.setCPUData(dat);

    dat[dim_y*(dim_x+2)+1] = 1;
    dat[dim_y*(dim_x+2)+1 + 1] = 2;
    dat[dim_y*(dim_x+2)+1 + 2] = 3;
    dat[dim_y*(dim_x+2)+1 + 3] = 4;
    dat[dim_y*(dim_x+2)+1 + 4] = 5;
    dat[dim_y*(dim_x+2)+1 + 5] = 6;

    // how many points only on the device and an OpenCL buffer for them
    int gpunum = (gpuDimX+2)*(gpuDimY+2);
    double *gpudat__host = new double[gpunum]{};

    cl::Buffer bufdat;

    try {

        bufdat = cl::Buffer(tau.cl_context, &gpudat__host[0], (&gpudat__host[gpunum-1])+1, false);

    } catch(cl::Error error) {
        std::cout << "[sample] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    tau.setGPUData(bufdat, gpuDimX, gpuDimY);

    tau.postCpuReceives();
    tau.postGpuReceives();

    /**********************************************/

    tau.startCpuTausch();
    tau.startGpuTausch();

    std::stringstream ss;
    ss << mpi_rank << " started halo exchange" << std::endl;
    EveryoneOutput(ss.str());

    tau.completeCpuTausch();
    tau.completeGpuTausch();

    std::stringstream ss2;
    ss2 << mpi_rank << " completed halo exchange" << std::endl;
    EveryoneOutput(ss2.str());

    delete[] dat;

}

void Sample::EveryoneOutput(const std::string &inMessage) {

    int myRank  = 0;
    int numProc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
    MPI_Comm_size(MPI_COMM_WORLD,&numProc);
    for(int iRank = 0; iRank < numProc; ++iRank){
        if(myRank == iRank)
            std::cout << inMessage;
        MPI_Barrier(MPI_COMM_WORLD);
    }

}

