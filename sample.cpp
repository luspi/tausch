#include "sample.h"

/*****************************
 *
 * This code uses the following test setup
 *
 *     |             20              |
 *   -----------------------------------
 *     |   CPU                 3     |
 *     |           -------------     |
 *     |           |    GPU    |     |
 *  2  |           |           |     |
 *  0  |     8     -------------  5  |
 *     |                             |
 *     |                      10     |
 *     |                             |
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

    Tausch tau(dim_x, dim_y);

    // the width of the halos
    int halowidth = 1;

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dim_x+2)*(dim_y+2);
    double *dat = new double[num]{};

    tau.setCPUHaloInfo(halowidth, dat);

    dat[dim_y*(dim_x+2)+1] = 1;
    dat[dim_y*(dim_x+2)+1 + 1] = 2;
    dat[dim_y*(dim_x+2)+1 + 2] = 3;
    dat[dim_y*(dim_x+2)+1 + 3] = 4;
    dat[dim_y*(dim_x+2)+1 + 4] = 5;
    dat[dim_y*(dim_x+2)+1 + 5] = 6;

    // how many points only on the device and an OpenCL buffer for them
    int gpunum = (7+2)*(7+2);
    double *gpudat__host = new double[gpunum]{};

    cl::Buffer bufdat(tau.cl_context, &gpudat__host[0], (&gpudat__host[gpunum-1])+1, false);

    // the coords of the GPU chunk (bottom left corner (x,y) and top right corner (x,y))
    int *coords = new int[4]{8,10,15,17};

    tau.setGPUHaloInfo(halowidth, bufdat, coords);

    /**********************************************/

    tau.startTausch();

    std::cout << mpi_rank << " started halo exchange" << std::endl;

    tau.completeTausch();

    std::cout << mpi_rank << " completed halo exchange" << std::endl;

    delete[] dat;

}

