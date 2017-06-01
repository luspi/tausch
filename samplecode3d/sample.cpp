#include "sample.h"

Sample::Sample(int *localDim, int loops, int *cpuHaloWidth, int *mpiNum) {

    for(unsigned int i = 0; i < 3; ++i)
        this->localDim[i] = localDim[i];
    this->loops = loops;
    for(int i = 0; i < 6; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    for(unsigned int i = 0; i < 3; ++i)
        this->mpiNum[i] = mpiNum[i];

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1, right = mpiRank+1;
    top = mpiRank+mpiNum[TAUSCH_X], bottom = mpiRank-mpiNum[TAUSCH_X];
    front = mpiRank-mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]; back = mpiRank+mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];

    if(mpiRank%mpiNum[TAUSCH_X] == 0)
        left += mpiNum[TAUSCH_X];
    if((mpiRank+1)%mpiNum[TAUSCH_X] == 0)
        right -= mpiNum[TAUSCH_X];
    if(mpiRank%(mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]) >= (mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]-mpiNum[TAUSCH_X]))
        top -= mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];
    if(mpiRank%(mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]) < mpiNum[TAUSCH_X])
        bottom += mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];
    if(mpiRank < mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y])
        front += mpiSize;
    if(mpiRank >= mpiSize-mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y])
        back -= mpiSize;

    numBuffers = 2;
    valuesPerPoint = 1;
    dat1 = new double[valuesPerPoint*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*
                                     (localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])*
                                     (localDim[2] + cpuHaloWidth[4] + cpuHaloWidth[5])]{};
    dat2 = new double[valuesPerPoint*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*
                                     (localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])*
                                     (localDim[2] + cpuHaloWidth[4] + cpuHaloWidth[5])]{};

    for(int k = 0; k < localDim[TAUSCH_Z]; ++k)
        for(int j = 0; j < localDim[TAUSCH_Y]; ++j)
            for(int i = 0; i < localDim[TAUSCH_X]; ++i) {
                for(int val = 0; val < valuesPerPoint; ++val) {
                    dat1[valuesPerPoint*((k+cpuHaloWidth[4])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                         (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                         (j+cpuHaloWidth[3])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) +
                                          i+cpuHaloWidth[0]) + val] = (k*localDim[TAUSCH_X]*localDim[TAUSCH_Y] + j*localDim[TAUSCH_X] + i + 1)*10+val;
                    dat2[valuesPerPoint*((k+cpuHaloWidth[4])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                         (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                         (j+cpuHaloWidth[3])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) +
                                          i+cpuHaloWidth[0])+val] = (5+k*localDim[TAUSCH_X]*localDim[TAUSCH_Y] + j*localDim[TAUSCH_X] + i + 1)*10+val;
                }
            }

    int tauschLocalDim[3] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1],
                             localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3],
                             localDim[2]+cpuHaloWidth[4]+cpuHaloWidth[5]};
    tausch = new Tausch3D<double>(tauschLocalDim, MPI_DOUBLE, numBuffers, valuesPerPoint);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new int*[6];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new int*[6];

    localHaloSpecs[0] = new int[8]{cpuHaloWidth[0], 0, 0,
                                   cpuHaloWidth[1], cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   left, 0};
    localHaloSpecs[1] = new int[8]{localDim[0], 0, 0,
                                   cpuHaloWidth[0], cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   right, 1};
    localHaloSpecs[2] = new int[8]{0, localDim[1], 0,
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[3], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   top, 2};
    localHaloSpecs[3] = new int[8]{0, cpuHaloWidth[3], 0,
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[2], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   bottom, 3};
    localHaloSpecs[4] = new int[8]{0, 0, cpuHaloWidth[4],
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3], cpuHaloWidth[5],
                                   front, 4};
    localHaloSpecs[5] = new int[8]{0, 0, localDim[2],
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3], cpuHaloWidth[4],
                                   back, 5};

    remoteHaloSpecs[0] = new int[8]{0, 0, 0,
                                   cpuHaloWidth[0], cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   left, 1};
    remoteHaloSpecs[1] = new int[8]{localDim[0]+cpuHaloWidth[0], 0, 0,
                                   cpuHaloWidth[1], cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   right, 0};
    remoteHaloSpecs[2] = new int[8]{0, localDim[1]+cpuHaloWidth[3], 0,
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[2], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   top, 3};
    remoteHaloSpecs[3] = new int[8]{0, 0, 0,
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[3], cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5],
                                   bottom, 2};
    remoteHaloSpecs[4] = new int[8]{0, 0, 0,
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3], cpuHaloWidth[4],
                                   front, 5};
    remoteHaloSpecs[5] = new int[8]{0, 0, localDim[2]+cpuHaloWidth[4],
                                   cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1], cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3], cpuHaloWidth[5],
                                   back, 4};

    tausch->setLocalHaloInfoCpu(6, localHaloSpecs);
    tausch->setRemoteHaloInfoCpu(6, remoteHaloSpecs);

}

Sample::~Sample() {

    for(int i = 0; i < 6; ++i) {
        delete[] localHaloSpecs[i];
        delete[] remoteHaloSpecs[i];
    }
    delete[] localHaloSpecs;
    delete[] remoteHaloSpecs;
    delete tausch;
    delete[] dat1;
    delete[] dat2;

}

void Sample::launchCPU() {

    int sendtags[6] = {0, 1, 2, 3, 4, 5};
    int recvtags[6] = {1, 0, 3, 2, 5, 4};

    for(int iter = 0; iter < loops; ++iter) {

        tausch->postAllReceivesCpu(recvtags);

        // left/right

        tausch->packNextSendBufferCpu(0, dat1);
        tausch->packNextSendBufferCpu(0, dat2);
        tausch->sendCpu(0, sendtags[0]);

        tausch->packNextSendBufferCpu(1, dat1);
        tausch->packNextSendBufferCpu(1, dat2);
        tausch->sendCpu(1, sendtags[1]);

        tausch->recvCpu(0);
        tausch->unpackNextRecvBufferCpu(0, dat1);
        tausch->unpackNextRecvBufferCpu(0, dat2);

        tausch->recvCpu(1);
        tausch->unpackNextRecvBufferCpu(1, dat1);
        tausch->unpackNextRecvBufferCpu(1, dat2);

        // top/bottom

        tausch->packNextSendBufferCpu(2, dat1);
        tausch->packNextSendBufferCpu(2, dat2);
        tausch->sendCpu(2, sendtags[2]);

        tausch->packNextSendBufferCpu(3, dat1);
        tausch->packNextSendBufferCpu(3, dat2);
        tausch->sendCpu(3, sendtags[3]);

        tausch->recvCpu(2);
        tausch->unpackNextRecvBufferCpu(2, dat1);
        tausch->unpackNextRecvBufferCpu(2, dat2);

        tausch->recvCpu(3);
        tausch->unpackNextRecvBufferCpu(3, dat1);
        tausch->unpackNextRecvBufferCpu(3, dat2);

        // front/back

        tausch->packNextSendBufferCpu(4, dat1);
        tausch->packNextSendBufferCpu(4, dat2);
        tausch->sendCpu(4, sendtags[4]);

        tausch->packNextSendBufferCpu(5, dat1);
        tausch->packNextSendBufferCpu(5, dat2);
        tausch->sendCpu(5, sendtags[5]);

        tausch->recvCpu(4);
        tausch->unpackNextRecvBufferCpu(4, dat1);
        tausch->unpackNextRecvBufferCpu(4, dat2);

        tausch->recvCpu(5);
        tausch->unpackNextRecvBufferCpu(5, dat1);
        tausch->unpackNextRecvBufferCpu(5, dat2);

    }

}

void Sample::print() {

    for(int z = 0; z < localDim[TAUSCH_Z]+cpuHaloWidth[4]+cpuHaloWidth[5]; ++z) {

        std::cout << std::endl << "z = " << z << std::endl;

        for(int j = localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {

            for(int val = 0; val < valuesPerPoint; ++val) {
                for(int i = 0; i < localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << dat1[z*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                                        (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                                      j*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i] << " ";
                if(val != valuesPerPoint-1)
                    std::cout << "   ";
            }
            std::cout << "          ";
            for(int val = 0; val < valuesPerPoint; ++val) {
                for(int i = 0; i < localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << dat2[z*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                                        (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                                      j*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i] << " ";
                if(val != valuesPerPoint-1)
                    std::cout << "   ";
            }
            std::cout << std::endl;
        }

    }

}
