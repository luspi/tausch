#include <tausch/tausch.h>
#include <cmath>
#include <chrono>

int main(int argc, char** argv) {

    // Initialize MPI.
    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided);

    // Get MPI metadata
    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // The local dimensions of the domain
    int localDim[2];
    localDim[TAUSCH_X] = 100;
    localDim[TAUSCH_Y] = 120;

    // The width of the halos along the four edges
    int cpuHaloWidth[4];
    cpuHaloWidth[TAUSCH_LEFT] = 1;
    cpuHaloWidth[TAUSCH_RIGHT] = 4;
    cpuHaloWidth[TAUSCH_TOP] = 2;
    cpuHaloWidth[TAUSCH_BOTTOM] = 1;

    // The layout of the MPI ranks. If mpiSize is a perfect square we take its square root for each dimension,
    // otherwise all MPI ranks are lined up in the x direction
    int mpiNum[2];
    mpiNum[TAUSCH_X] = std::sqrt(mpiSize);
    mpiNum[TAUSCH_Y] = std::sqrt(mpiSize);
    if(mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y] != mpiSize) {
        mpiNum[TAUSCH_X] = mpiSize;
        mpiNum[TAUSCH_Y] = 1;
    }

    // The four MPI rank neighbours we use for this example
    int left = mpiRank-1;
    int right = mpiRank+1;
    // If we are along a domain boundary, we wrap around to the opposite end (periodic)
    if(mpiRank%mpiNum[TAUSCH_X] == 0)          left += mpiNum[TAUSCH_X];
    if((mpiRank+1)%mpiNum[TAUSCH_X] == 0)      right -= mpiNum[TAUSCH_X];

    //. Here we will use two buffers that require a halo exchange
    int numBuffers = 2;
    double *dat1 = new double[(localDim[TAUSCH_X] + cpuHaloWidth[TAUSCH_LEFT] + cpuHaloWidth[TAUSCH_RIGHT])*
                              (localDim[TAUSCH_Y] + cpuHaloWidth[TAUSCH_TOP] + cpuHaloWidth[TAUSCH_BOTTOM])]{};
    double *dat2 = new double[(localDim[TAUSCH_X] + cpuHaloWidth[TAUSCH_LEFT] + cpuHaloWidth[TAUSCH_RIGHT])*
                              (localDim[TAUSCH_Y] + cpuHaloWidth[TAUSCH_TOP] + cpuHaloWidth[TAUSCH_BOTTOM])]{};

    // We have four halo regions, one across each of the four edges
    int **remoteHaloSpecs = new int*[1];
    int **localHaloSpecs = new int*[1];

    // left edge (0) remote halo region: [x, y, w, h, receiver, tag]
    remoteHaloSpecs[0] = new int[6]{0, 0,
                                    cpuHaloWidth[TAUSCH_LEFT], cpuHaloWidth[TAUSCH_BOTTOM]+localDim[TAUSCH_Y]+cpuHaloWidth[TAUSCH_TOP],
                                    left, 0};

    // right edge (1) local halo region: [x, y, w, h, sender, tag]
    localHaloSpecs[0]  = new int[6]{localDim[TAUSCH_X], 0,
                                    cpuHaloWidth[TAUSCH_LEFT], cpuHaloWidth[TAUSCH_BOTTOM]+localDim[TAUSCH_Y]+cpuHaloWidth[TAUSCH_TOP],
                                    right, 0};

    // The Tausch object, using its double version. The pointer type is of type 'Tausch', although using Tausch2D directly would also be possible here
    Tausch2D<double> *tausch = new Tausch2D<double>(localDim, cpuHaloWidth, numBuffers, 1);

    // Tell Tausch about the local and remote halo regions
    tausch->setCpuLocalHaloInfo(1, localHaloSpecs);
    tausch->setCpuRemoteHaloInfo(1, remoteHaloSpecs);

    /*****************
     * HALO EXCHANGE *
     *****************/

    MPI_Barrier(MPI_COMM_WORLD);

    // Start a timer
    auto t_start = std::chrono::steady_clock::now();

    // post the MPI receives
    tausch->postMpiReceives();

    // pack the right buffers and send them off
    tausch->packNextSendBuffer(0, dat1);
    tausch->packNextSendBuffer(0, dat2);
    tausch->send(0);

    // receive the left buffers and unpack them
    tausch->recv(0);
    tausch->unpackNextRecvBuffer(0, dat1);
    tausch->unpackNextRecvBuffer(0, dat2);

    MPI_Barrier(MPI_COMM_WORLD);

    /*****************/

    // End timer
    auto t_end = std::chrono::steady_clock::now();

    // Output timing result
    if(mpiRank == 0)
        std::cout << "Required time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    // Clean up memory
    delete[] localHaloSpecs[0];
    delete[] remoteHaloSpecs[0];
    delete[] localHaloSpecs;
    delete[] remoteHaloSpecs;
    delete[] dat1;
    delete[] dat2;
    delete tausch;

    MPI_Finalize();
    return 0;
}
