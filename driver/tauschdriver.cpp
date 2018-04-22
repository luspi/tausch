#include "tauschdriver.h"

TauschDriver::TauschDriver(size_t *localDim, int iterations, int *mpiNum) {


    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
    MPI_Comm_size(MPI_COMM_WORLD,&numProc);

    this->localDim[0] = localDim[0];
    this->localDim[1] = localDim[1];
    this->iterations = iterations;
    this->mpiNum[0] = mpiNum[0];
    this->mpiNum[1] = mpiNum[1];

    deltaX = 1.0/localDim[0];
    deltaY = 1.0/localDim[1];

    size_t valuesPerPointPerBuffer[2] = {1,5};

    tausch = new Tausch<double>(MPI_DOUBLE, 2, valuesPerPointPerBuffer, MPI_COMM_WORLD);

    left = myRank-1, right = myRank+1, top = myRank+mpiNum[0], bottom = myRank-mpiNum[0];

    if(myRank%mpiNum[0] == 0)
        left += mpiNum[0];
    if((myRank+1)%mpiNum[0] == 0)
        right -= mpiNum[0];

    if(myRank < mpiNum[0])
        bottom += numProc;
    bottomleft = left-mpiNum[0];
    if(myRank < mpiNum[0])
        bottomleft = left+mpiNum[0]*(mpiNum[1]-1);
    bottomright = right-mpiNum[0];
    if(myRank < mpiNum[0])
        bottomright = right+mpiNum[0]*(mpiNum[1]-1);

    if(myRank >= numProc-mpiNum[0])
        top -= numProc;
    topleft = left+mpiNum[0];
    if(myRank >= numProc-mpiNum[0])
        topleft = left-mpiNum[0]*(mpiNum[1]-1);
    topright = right+mpiNum[0];
    if(myRank >= numProc-mpiNum[0])
        topright = right-mpiNum[0]*(mpiNum[1]-1);

    if(myRank == 0)
        bottomleft = numProc-1;
    if(myRank == mpiNum[0]-1)
        bottomright = numProc-mpiNum[0];
    if(myRank == numProc-mpiNum[0])
        topleft = mpiNum[0]-1;
    if(myRank == numProc-1)
        topright = 0;

    int stencilNumPoints = 5;

    cpuData = new double[(1+localDim[0]+1)*(1+localDim[1]+1)]{};
    cpuStencil = new double[stencilNumPoints*(1+localDim[0]+1)*(1+localDim[1]+1)]{};

    for(size_t j = 0; j < 1+localDim[1]+1; ++j) {
        for(size_t i = 0; i < 1+localDim[0]+1; ++i) {
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 0] = -1*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 1] = -1*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 2] = -1*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 3] = 8*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 4] = -1*((deltaX*deltaY)/(6.0));
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);
    for(size_t j = 0; j < localDim[1]; ++j)
        for(size_t i = 0; i < localDim[0]; ++i)
            cpuData[(j+1)*(1+localDim[0]+1) + i+1] = dis(gen);

    size_t tauschLocalDim[2] = {1+localDim[0]+1, 1+localDim[1]+1};

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new TauschHaloSpec[4];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new TauschHaloSpec[4];

    localHaloSpecs[0].bufferWidth = tauschLocalDim[0]; localHaloSpecs[0].bufferHeight = tauschLocalDim[1];
    localHaloSpecs[0].haloX = 1; localHaloSpecs[0].haloY = 0;
    localHaloSpecs[0].haloWidth = 1; localHaloSpecs[0].haloHeight = 1+localDim[1]+1;
    localHaloSpecs[0].remoteMpiRank = left;
    remoteHaloSpecs[0].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[0].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecs[0].haloX = 0; remoteHaloSpecs[0].haloY = 0;
    remoteHaloSpecs[0].haloWidth = 1; remoteHaloSpecs[0].haloHeight = 1+localDim[1]+1;
    remoteHaloSpecs[0].remoteMpiRank = left;

    localHaloSpecs[1].bufferWidth = tauschLocalDim[0]; localHaloSpecs[1].bufferHeight = tauschLocalDim[1];
    localHaloSpecs[1].haloX = localDim[0]; localHaloSpecs[1].haloY = 0;
    localHaloSpecs[1].haloWidth = 1; localHaloSpecs[1].haloHeight = 1+localDim[1]+1;
    localHaloSpecs[1].remoteMpiRank = right;
    remoteHaloSpecs[1].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[1].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecs[1].haloX = 1+localDim[0]; remoteHaloSpecs[1].haloY = 0;
    remoteHaloSpecs[1].haloWidth = 1; remoteHaloSpecs[1].haloHeight = 1+localDim[1]+1;
    remoteHaloSpecs[1].remoteMpiRank = right;

    localHaloSpecs[2].bufferWidth = tauschLocalDim[0]; localHaloSpecs[2].bufferHeight = tauschLocalDim[1];
    localHaloSpecs[2].haloX = 0; localHaloSpecs[2].haloY = localDim[1];
    localHaloSpecs[2].haloWidth = 1+localDim[0]+1; localHaloSpecs[2].haloHeight = 1;
    localHaloSpecs[2].remoteMpiRank = top;
    remoteHaloSpecs[2].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[2].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecs[2].haloX = 0; remoteHaloSpecs[2].haloY = 1+localDim[1];
    remoteHaloSpecs[2].haloWidth = 1+localDim[0]+1; remoteHaloSpecs[2].haloHeight = 1;
    remoteHaloSpecs[2].remoteMpiRank = top;

    localHaloSpecs[3].bufferWidth = tauschLocalDim[0]; localHaloSpecs[3].bufferHeight = tauschLocalDim[1];
    localHaloSpecs[3].haloX = 0; localHaloSpecs[3].haloY = 1;
    localHaloSpecs[3].haloWidth = 1+localDim[0]+1; localHaloSpecs[3].haloHeight = 1;
    localHaloSpecs[3].remoteMpiRank = bottom;
    remoteHaloSpecs[3].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[3].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecs[3].haloX = 0; remoteHaloSpecs[3].haloY = 0;
    remoteHaloSpecs[3].haloWidth = 1+localDim[0]+1; remoteHaloSpecs[3].haloHeight = 1;
    remoteHaloSpecs[3].remoteMpiRank = bottom;

    tausch->setLocalHaloInfo2D_CwC(4, localHaloSpecs);
    tausch->setRemoteHaloInfo2D_CwC(4, remoteHaloSpecs);

}

void TauschDriver::iterate() {

    int sendtagsCpu[4] = {0, 1, 2, 3};
    int recvtagsCpu[4] = {1, 0, 3, 2};

    for(int i = 0; i < iterations; ++i) {

        tausch->postAllReceives2D_CwC(recvtagsCpu);

        kernel(0,0,localDim[0], localDim[1]);

        tausch->packAndSend2D_CwC(2, cpuData, sendtagsCpu[2]);
        tausch->packAndSend2D_CwC(3, cpuData, sendtagsCpu[3]);

        tausch->recvAndUnpack2D_CwC(2, cpuData);
        tausch->recvAndUnpack2D_CwC(3, cpuData);

        tausch->packAndSend2D_CwC(0, cpuData, sendtagsCpu[0]);
        tausch->packAndSend2D_CwC(1, cpuData, sendtagsCpu[1]);

        tausch->recvAndUnpack2D_CwC(0, cpuData);
        tausch->recvAndUnpack2D_CwC(1, cpuData);

    }

}

void TauschDriver::kernel(int startX, int startY, int endX, int endY) {

    int onerow = (1+localDim[0]+1);

    for(int j = startY; j < endY; ++j) {
        for(int i = startX; i < endX; ++i) {

            int index = (j+1)*onerow+i+1;

            cpuData[index] = cpuStencil[5*index] * cpuData[index-onerow-1] +
                             cpuStencil[5*index+1] * cpuData[index-onerow] +
                             cpuStencil[5*index+2] * cpuData[index-1] +
                             cpuStencil[5*index+3] * cpuData[index] +
                             cpuStencil[5*index+4] * cpuData[index+onerow-1] +

                             cpuStencil[5*index + 5] * cpuData[index-onerow+1] +
                             cpuStencil[5*index + 5+2] * cpuData[index+1] +
                             cpuStencil[5*index + 5+4] * cpuData[index+onerow+1] +

                             cpuStencil[5*(index+onerow) + 1] * cpuData[index+onerow];

        }

    }

}

TauschDriver::~TauschDriver() {
    delete tausch;
}
