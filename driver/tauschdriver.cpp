#include "tauschdriver.h"

TauschDriver::TauschDriver(int *localDim, int iterations, int *mpiNum) {

    // Get MPI info
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
    MPI_Comm_size(MPI_COMM_WORLD,&numProc);

    // store meta info
    this->localDim[0] = localDim[0];
    this->localDim[1] = localDim[1];
    this->iterations = iterations;
    this->mpiNum[0] = mpiNum[0];
    this->mpiNum[1] = mpiNum[1];

    // The first buffer is the data (one data point per mesh point)
    // The second buffer is the stencil (5 data points per mesh point)
    size_t valuesPerPointPerBuffer[2] = {1,5};
    // The tausch object
    tausch = new Tausch<double>(MPI_DOUBLE, 2, valuesPerPointPerBuffer, MPI_COMM_WORLD);

    // figure out which rank is at each side
    left = myRank-1, right = myRank+1, top = myRank+mpiNum[0], bottom = myRank-mpiNum[0];
    if(myRank%mpiNum[0] == 0)
        left += mpiNum[0];
    if((myRank+1)%mpiNum[0] == 0)
        right -= mpiNum[0];
    if(myRank < mpiNum[0])
        bottom += numProc;
    if(myRank >= numProc-mpiNum[0])
        top -= numProc;

    int stencilNumPoints = 5;
    // The two buffer for the data and the stencil
    cpuData = new double[(1+localDim[0]+1)*(1+localDim[1]+1)]{};
    cpuStencil = new double[stencilNumPoints*(1+localDim[0]+1)*(1+localDim[1]+1)]{};

    // Fill the data buffer with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);
    for(int j = 0; j < localDim[1]; ++j)
        for(int i = 0; i < localDim[0]; ++i)
            cpuData[(j+1)*(1+localDim[0]+1) + i+1] = dis(gen);

    // Fill the stencil data with the stencil (5 values stored for each point, the remaining 4 are taken from the stencils of other points around)
    double deltaX = 1.0/localDim[0];
    double deltaY = 1.0/localDim[1];
    for(int j = 0; j < 1+localDim[1]+1; ++j) {
        for(int i = 0; i < 1+localDim[0]+1; ++i) {
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 0] = -1*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 1] = -1*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 2] = -1*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 3] = 8*((deltaX*deltaY)/(6.0));
            cpuStencil[stencilNumPoints*(j*(1+localDim[0]+1) + i) + 4] = -1*((deltaX*deltaY)/(6.0));
        }
    }

    // These are the 4 remote halos that are needed by this rank
    remoteHaloSpecs = new TauschHaloSpec[4];
    // These are the 4 local halos that are needed to be sent by this rank
    localHaloSpecs = new TauschHaloSpec[4];

    localHaloSpecs[0].bufferWidth = 1+localDim[0]+1; localHaloSpecs[0].bufferHeight = 1+localDim[1]+1;
    localHaloSpecs[0].haloX = 1; localHaloSpecs[0].haloY = 0;
    localHaloSpecs[0].haloWidth = 1; localHaloSpecs[0].haloHeight = 1+localDim[1]+1;
    localHaloSpecs[0].remoteMpiRank = left;
    remoteHaloSpecs[0].bufferWidth = 1+localDim[0]+1; remoteHaloSpecs[0].bufferHeight = 1+localDim[1]+1;
    remoteHaloSpecs[0].haloX = 0; remoteHaloSpecs[0].haloY = 0;
    remoteHaloSpecs[0].haloWidth = 1; remoteHaloSpecs[0].haloHeight = 1+localDim[1]+1;
    remoteHaloSpecs[0].remoteMpiRank = left;

    localHaloSpecs[1].bufferWidth = 1+localDim[0]+1; localHaloSpecs[1].bufferHeight = 1+localDim[1]+1;
    localHaloSpecs[1].haloX = localDim[0]; localHaloSpecs[1].haloY = 0;
    localHaloSpecs[1].haloWidth = 1; localHaloSpecs[1].haloHeight = 1+localDim[1]+1;
    localHaloSpecs[1].remoteMpiRank = right;
    remoteHaloSpecs[1].bufferWidth = 1+localDim[0]+1; remoteHaloSpecs[1].bufferHeight = 1+localDim[1]+1;
    remoteHaloSpecs[1].haloX = 1+localDim[0]; remoteHaloSpecs[1].haloY = 0;
    remoteHaloSpecs[1].haloWidth = 1; remoteHaloSpecs[1].haloHeight = 1+localDim[1]+1;
    remoteHaloSpecs[1].remoteMpiRank = right;

    localHaloSpecs[2].bufferWidth = 1+localDim[0]+1; localHaloSpecs[2].bufferHeight = 1+localDim[1]+1;
    localHaloSpecs[2].haloX = 0; localHaloSpecs[2].haloY = localDim[1];
    localHaloSpecs[2].haloWidth = 1+localDim[0]+1; localHaloSpecs[2].haloHeight = 1;
    localHaloSpecs[2].remoteMpiRank = top;
    remoteHaloSpecs[2].bufferWidth = 1+localDim[0]+1; remoteHaloSpecs[2].bufferHeight = 1+localDim[1]+1;
    remoteHaloSpecs[2].haloX = 0; remoteHaloSpecs[2].haloY = 1+localDim[1];
    remoteHaloSpecs[2].haloWidth = 1+localDim[0]+1; remoteHaloSpecs[2].haloHeight = 1;
    remoteHaloSpecs[2].remoteMpiRank = top;

    localHaloSpecs[3].bufferWidth = 1+localDim[0]+1; localHaloSpecs[3].bufferHeight = 1+localDim[1]+1;
    localHaloSpecs[3].haloX = 0; localHaloSpecs[3].haloY = 1;
    localHaloSpecs[3].haloWidth = 1+localDim[0]+1; localHaloSpecs[3].haloHeight = 1;
    localHaloSpecs[3].remoteMpiRank = bottom;
    remoteHaloSpecs[3].bufferWidth = 1+localDim[0]+1; remoteHaloSpecs[3].bufferHeight = 1+localDim[1]+1;
    remoteHaloSpecs[3].haloX = 0; remoteHaloSpecs[3].haloY = 0;
    remoteHaloSpecs[3].haloWidth = 1+localDim[0]+1; remoteHaloSpecs[3].haloHeight = 1;
    remoteHaloSpecs[3].remoteMpiRank = bottom;

    tausch->setLocalHaloInfo2D_CwC(4, localHaloSpecs);
    tausch->setRemoteHaloInfo2D_CwC(4, remoteHaloSpecs);

}

// This function performs all requested iterations
void TauschDriver::iterate() {

    // The MPI tags
    int sendtagsCpu[4] = {0, 1, 2, 3};
    int recvtagsCpu[4] = {1, 0, 3, 2};

    for(int i = 0; i < iterations; ++i) {

        // post all recvs
        tausch->postAllReceives2D_CwC(recvtagsCpu);

        // Apply stencil (9-point stencil)
        kernel(0,0,localDim[0], localDim[1]);

        // Pack and send top and bottom halos
        tausch->packSendBuffer2D_CwC(2, 0, cpuData);
        tausch->packSendBuffer2D_CwC(2, 1, cpuStencil);
        tausch->send2D_CwC(2, sendtagsCpu[2]);
        tausch->packSendBuffer2D_CwC(3, 0, cpuData);
        tausch->packSendBuffer2D_CwC(3, 1, cpuStencil);
        tausch->send2D_CwC(3, sendtagsCpu[3]);

        // Recv and unpack top and bottom halos
        tausch->recv2D_CwC(2);
        tausch->unpackRecvBuffer2D_CwC(2, 0, cpuData);
        tausch->unpackRecvBuffer2D_CwC(2, 1, cpuStencil);
        tausch->recv2D_CwC(3);
        tausch->unpackRecvBuffer2D_CwC(3, 0, cpuData);
        tausch->unpackRecvBuffer2D_CwC(3, 1, cpuStencil);

        // Pack and send left and right halos
        tausch->packSendBuffer2D_CwC(0, 0, cpuData);
        tausch->packSendBuffer2D_CwC(0, 1, cpuStencil);
        tausch->send2D_CwC(0, sendtagsCpu[0]);
        tausch->packSendBuffer2D_CwC(1, 0, cpuData);
        tausch->packSendBuffer2D_CwC(1, 1, cpuStencil);
        tausch->send2D_CwC(1, sendtagsCpu[1]);

        // Recv and unpack left and right halos
        tausch->recv2D_CwC(0);
        tausch->unpackRecvBuffer2D_CwC(0, 0, cpuData);
        tausch->unpackRecvBuffer2D_CwC(0, 1, cpuStencil);
        tausch->recv2D_CwC(1);
        tausch->unpackRecvBuffer2D_CwC(1, 0, cpuData);
        tausch->unpackRecvBuffer2D_CwC(1, 1, cpuStencil);

    }

}

// This function applies a 9-point stencil (serial version)
void TauschDriver::kernel(int startX, int startY, int endX, int endY) {

    int onerow = (1+localDim[0]+1);

    for(int j = startY; j < endY; ++j) {

        // The final +1 is taken from the calc of index below. As this is a fixed 1, we can add it on here already
        int index_p1 = (j+1)*onerow  +1;

        for(int i = startX; i < endX; ++i) {

            int index = index_p1+i;

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
