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
    localDim[TX] = 100;
    localDim[TY] = 120;

    // The width of the halos along the four edges
    int cpuHaloWidth[4];
    cpuHaloWidth[TLEFT] = 1;
    cpuHaloWidth[TRIGHT] = 4;
    cpuHaloWidth[TTOP] = 2;
    cpuHaloWidth[TBOTTOM] = 1;

    // The layout of the MPI ranks. If mpiSize is a perfect square we take its square root for each dimension,
    // otherwise all MPI ranks are lined up in the x direction
    int mpiNum[2];
    mpiNum[TX] = std::sqrt(mpiSize);
    mpiNum[TY] = std::sqrt(mpiSize);
    if(mpiNum[TX]*mpiNum[TY] != mpiSize) {
        mpiNum[TX] = mpiSize;
        mpiNum[TY] = 1;
    }

    // The four MPI rank neighbours we use for this example
    int left = mpiRank-1;
    int right = mpiRank+1;
    int top = mpiRank+mpiNum[TX];
    int bottom = mpiRank-mpiNum[TX];
    // If we are along a domain boundary, we wrap around to the opposite end (periodic)
    if(mpiRank%mpiNum[TX] == 0)          left += mpiNum[TX];
    if((mpiRank+1)%mpiNum[TX] == 0)      right -= mpiNum[TX];
    if(mpiRank < mpiNum[TX])             bottom += mpiSize;
    if(mpiRank >= mpiSize-mpiNum[TX])    top -= mpiSize;

    //. Here we will use two buffers that require a halo exchange
    int numBuffers = 2;
    double *dat1 = new double[(localDim[TX] + cpuHaloWidth[TLEFT] + cpuHaloWidth[TRIGHT])*
                              (localDim[TY] + cpuHaloWidth[TTOP] + cpuHaloWidth[TBOTTOM])]{};
    double *dat2 = new double[(localDim[TX] + cpuHaloWidth[TLEFT] + cpuHaloWidth[TRIGHT])*
                              (localDim[TY] + cpuHaloWidth[TTOP] + cpuHaloWidth[TBOTTOM])]{};

    // The Tausch object, using its double version. The pointer type is of type 'Tausch', although Tausch2D owuld also be possible here.
    Tausch2D<double> *tausch = new Tausch2D<double>(localDim, cpuHaloWidth, numBuffers, 1);

    // We have four halo regions, one across each of the four edges
    int **remoteHaloSpecs = new int*[4];
    int **localHaloSpecs = new int*[4];

    // left edge (0) local and remote halo region: [x, y, w, h, sender/receiver, tag]
    localHaloSpecs[0]  = new int[6]{cpuHaloWidth[TLEFT], 0,
                                    cpuHaloWidth[TRIGHT], cpuHaloWidth[TBOTTOM]+localDim[TY]+cpuHaloWidth[TTOP],
                                    left, 0};
    remoteHaloSpecs[0] = new int[6]{0, 0,
                                    cpuHaloWidth[TLEFT], cpuHaloWidth[TBOTTOM]+localDim[TY]+cpuHaloWidth[TTOP],
                                    left, 1};
    // right edge (1) local and remote halo region: [x, y, w, h, sender/receiver, tag]
    localHaloSpecs[1]  = new int[6]{localDim[TX], 0,
                                    cpuHaloWidth[TLEFT], cpuHaloWidth[TBOTTOM]+localDim[TY]+cpuHaloWidth[TTOP],
                                    right, 1};
    remoteHaloSpecs[1] = new int[6]{cpuHaloWidth[TLEFT]+localDim[TX], 0,
                                    cpuHaloWidth[TRIGHT], cpuHaloWidth[TBOTTOM]+localDim[TY]+cpuHaloWidth[TTOP],
                                    right, 0};
    // top edge (2) local and remote halo region: [x, y, w, h, sender/receiver, tag]
    localHaloSpecs[2]  = new int[6]{0, localDim[TY],
                                    cpuHaloWidth[TLEFT]+localDim[TX]+cpuHaloWidth[TRIGHT], cpuHaloWidth[TBOTTOM],
                                    top, 2};
    remoteHaloSpecs[2] = new int[6]{0, cpuHaloWidth[TBOTTOM]+localDim[TY],
                                    cpuHaloWidth[TLEFT]+localDim[TX]+cpuHaloWidth[TRIGHT], cpuHaloWidth[TTOP],
                                    top, 3};
    // bottom edge (3) local and remote halo region: [x, y, w, h, sender/receiver, tag]
    localHaloSpecs[3]  = new int[6]{0, cpuHaloWidth[TBOTTOM],
                                    cpuHaloWidth[TLEFT]+localDim[TX]+cpuHaloWidth[TRIGHT], cpuHaloWidth[TTOP],
                                    bottom, 3};
    remoteHaloSpecs[3] = new int[6]{0, 0,
                                    cpuHaloWidth[TLEFT]+localDim[TX]+cpuHaloWidth[TRIGHT], cpuHaloWidth[TBOTTOM],
                                    bottom, 2};

    // Tell Tausch about the local and remote halo regions
    tausch->setCpuLocalHaloInfo(4, localHaloSpecs);
    tausch->setCpuRemoteHaloInfo(4, remoteHaloSpecs);

    MPI_Barrier(MPI_COMM_WORLD);

    // Start a timer
    auto t_start = std::chrono::steady_clock::now();

    // post the MPI receives
    tausch->postMpiReceives();

    // diagonal trick: first communicate left/right ...

    // pack the left buffers and send them off
    tausch->packNextSendBuffer(0, dat1);
    tausch->packNextSendBuffer(0, dat2);
    tausch->send(0);
    // pack the right buffers and send them off
    tausch->packNextSendBuffer(1, dat1);
    tausch->packNextSendBuffer(1, dat2);
    tausch->send(1);
    // receive the left buffers and unpack them
    tausch->recv(0);
    tausch->unpackNextRecvBuffer(0, dat1);
    tausch->unpackNextRecvBuffer(0, dat2);
    // receive the right buffers and unpack them
    tausch->recv(1);
    tausch->unpackNextRecvBuffer(1, dat1);
    tausch->unpackNextRecvBuffer(1, dat2);

    // ... then communicate top/bottom

    // pack the top buffers and send them off
    tausch->packNextSendBuffer(2, dat1);
    tausch->packNextSendBuffer(2, dat2);
    tausch->send(2);
    // pack the bottom buffers and send them off
    tausch->packNextSendBuffer(3, dat1);
    tausch->packNextSendBuffer(3, dat2);
    tausch->send(3);
    // receive the top buffers and unpack them
    tausch->recv(2);
    tausch->unpackNextRecvBuffer(2, dat1);
    tausch->unpackNextRecvBuffer(2, dat2);
    // receive the bottom buffers and unpack them
    tausch->recv(3);
    tausch->unpackNextRecvBuffer(3, dat1);
    tausch->unpackNextRecvBuffer(3, dat2);

    MPI_Barrier(MPI_COMM_WORLD);

    // End timer
    auto t_end = std::chrono::steady_clock::now();

    // Output timing result
    if(mpiRank == 0)
        std::cout << "Required time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    // Clean up memory
    for(int i = 0; i < 4; ++i) {
        delete[] localHaloSpecs[i];
        delete[] remoteHaloSpecs[i];
    }
    delete[] localHaloSpecs;
    delete[] remoteHaloSpecs;
    delete[] dat1;
    delete[] dat2;
    delete tausch;

    MPI_Finalize();
    return 0;
}
