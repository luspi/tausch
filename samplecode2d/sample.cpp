#include "sample.h"

Sample::Sample(size_t *localDim, size_t *gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, int *mpiNum, bool buildlog, bool hybrid, bool gpuonly) {

    this->localDim[0] = localDim[0];
    this->localDim[1] = localDim[1];
    this->loops = loops;
    for(int i = 0; i < 4; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    this->mpiNum[0] = mpiNum[0];
    this->mpiNum[1] = mpiNum[1];
#ifdef OPENCL
    this->hybrid = hybrid;
    this->gpuonly = gpuonly;
    this->gpuDim[0] = gpuDim[0];
    this->gpuDim[1] = gpuDim[1];
    for(int i = 0; i < 4; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];
    for(int i = 0; i < 4; ++i)
        this->cpuForGpuHaloWidth[i] = cpuForGpuHaloWidth[i];
#endif

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1; right = mpiRank+1; top = mpiRank+mpiNum[0]; bottom = mpiRank-mpiNum[0];
    if(mpiRank%mpiNum[0] == 0)
        left += mpiNum[0];
    if((mpiRank+1)%mpiNum[0] == 0)
        right -= mpiNum[0];
    if(mpiRank < mpiNum[0])
        bottom += mpiSize;
    if(mpiRank >= mpiSize-mpiNum[0])
        top -= mpiSize;

    // when changing this to anything but 1, re-adjust the launch() functions
    numBuffers = 1;

    valuesPerPointPerBuffer = new size_t[numBuffers];
    for(size_t b = 0; b < numBuffers; ++b)
        valuesPerPointPerBuffer[b] = 1;

    tausch = new Tausch<double>(MPI_DOUBLE, numBuffers, valuesPerPointPerBuffer, MPI_COMM_WORLD);

#ifdef OPENCL
    if(!gpuonly) {
#endif

        dat = new double*[numBuffers];
        for(size_t b = 0; b < numBuffers; ++b)
            dat[b] = new double[valuesPerPointPerBuffer[b]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*(localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])]();

        if(!hybrid) {
            for(size_t j = 0; j < localDim[1]; ++j)
                for(size_t i = 0; i < localDim[0]; ++i) {
                    for(size_t b = 0; b < numBuffers; ++b)
                        for(size_t val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                            dat[b][valuesPerPointPerBuffer[b]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                                    (b*5 + j*localDim[0]+i+1)*10+val;
                }
        } else {
            for(size_t j = 0; j < localDim[1]; ++j)
                for(size_t i = 0; i < localDim[0]; ++i) {
                    if(i >= (localDim[0]-gpuDim[0])/2 && i < (localDim[0]-gpuDim[0])/2+gpuDim[0] &&
                       j >= (localDim[1]-gpuDim[1])/2 && j < (localDim[1]-gpuDim[1])/2+gpuDim[1])
                        continue;
                    for(size_t b = 0; b < numBuffers; ++b)
                        for(size_t val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                            dat[b][valuesPerPointPerBuffer[b]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                                    (b*5 + j*localDim[0]+i+1)*10+val;
                }
        }

        size_t tauschLocalDim[2] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1], localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]};

        TauschHaloSpec remoteHaloSpecsCpu;
        TauschHaloSpec localHaloSpecsCpu;

        localHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpu.haloX = cpuHaloWidth[0]; localHaloSpecsCpu.haloY = 0;
        localHaloSpecsCpu.haloWidth = cpuHaloWidth[1]; localHaloSpecsCpu.haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
        localHaloSpecsCpu.remoteMpiRank = left;
        remoteHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpu.haloX = 0; remoteHaloSpecsCpu.haloY = 0;
        remoteHaloSpecsCpu.haloWidth = cpuHaloWidth[0]; remoteHaloSpecsCpu.haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
        remoteHaloSpecsCpu.remoteMpiRank = left;

        allLocalHaloIds[0] = size_t(tausch->addLocalHaloInfo2D_CwC(localHaloSpecsCpu));
        allRemoteHaloIds[0] = size_t(tausch->addRemoteHaloInfo2D_CwC(remoteHaloSpecsCpu));

        localHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpu.haloX = localDim[0]; localHaloSpecsCpu.haloY = 0;
        localHaloSpecsCpu.haloWidth = cpuHaloWidth[0]; localHaloSpecsCpu.haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
        localHaloSpecsCpu.remoteMpiRank = right;
        remoteHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpu.haloX = cpuHaloWidth[0]+localDim[0]; remoteHaloSpecsCpu.haloY = 0;
        remoteHaloSpecsCpu.haloWidth = cpuHaloWidth[1]; remoteHaloSpecsCpu.haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
        remoteHaloSpecsCpu.remoteMpiRank = right;

        allLocalHaloIds[1] = size_t(tausch->addLocalHaloInfo2D_CwC(localHaloSpecsCpu));
        allRemoteHaloIds[1] = size_t(tausch->addRemoteHaloInfo2D_CwC(remoteHaloSpecsCpu));

        localHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpu.haloX = 0; localHaloSpecsCpu.haloY = localDim[1];
        localHaloSpecsCpu.haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecsCpu.haloHeight = cpuHaloWidth[3];
        localHaloSpecsCpu.remoteMpiRank = top;
        remoteHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpu.haloX = 0; remoteHaloSpecsCpu.haloY = cpuHaloWidth[3]+localDim[1];
        remoteHaloSpecsCpu.haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecsCpu.haloHeight = cpuHaloWidth[2];
        remoteHaloSpecsCpu.remoteMpiRank = top;

        allLocalHaloIds[2] = size_t(tausch->addLocalHaloInfo2D_CwC(localHaloSpecsCpu));
        allRemoteHaloIds[2] = size_t(tausch->addRemoteHaloInfo2D_CwC(remoteHaloSpecsCpu));

        localHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpu.haloX = 0; localHaloSpecsCpu.haloY = cpuHaloWidth[3];
        localHaloSpecsCpu.haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecsCpu.haloHeight = cpuHaloWidth[2];
        localHaloSpecsCpu.remoteMpiRank = bottom;
        remoteHaloSpecsCpu.bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu.bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpu.haloX = 0; remoteHaloSpecsCpu.haloY = 0;
        remoteHaloSpecsCpu.haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecsCpu.haloHeight = cpuHaloWidth[3];
        remoteHaloSpecsCpu.remoteMpiRank = bottom;

        allLocalHaloIds[3] = size_t(tausch->addLocalHaloInfo2D_CwC(localHaloSpecsCpu));
        allRemoteHaloIds[3] = size_t(tausch->addRemoteHaloInfo2D_CwC(remoteHaloSpecsCpu));

#ifdef OPENCL
    }

    if(hybrid) {
        size_t tauschGpuDim[2] = {gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1], gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]};
        tausch->enableOpenCL2D(true, 64, true, buildlog);

        gpudat = new double*[numBuffers];
        for(size_t b = 0; b < numBuffers; ++b)
            gpudat[b] = new double[valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3])]{};

        for(size_t j = 0; j < gpuDim[1]; ++j)
            for(size_t i = 0; i < gpuDim[0]; ++i) {
                for(size_t b = 0; b < numBuffers; ++b)
                    for(size_t val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                        gpudat[b][valuesPerPointPerBuffer[b]*((j+gpuHaloWidth[3])*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i+gpuHaloWidth[0])+val] =
                                (b*5 + j*gpuDim[0]+i+1)*10+val;
            }

        try {
            cl_gpudat = new cl::Buffer[numBuffers];
            for(size_t b = 0; b < numBuffers; ++b) {
                size_t s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
                cl_gpudat[b] = cl::Buffer(tausch->getOpenCLContext2D(), &gpudat[b][0], (&gpudat[b][s-1])+1, false);
            }
        } catch(cl::Error error) {
            std::cerr << "Samplecode2D :: constructor :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

        size_t tauschLocalDim[2] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1], localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]};

        remoteHaloSpecsCpuForGpu = new TauschHaloSpec[4];
        localHaloSpecsCpuForGpu = new TauschHaloSpec[4];

        remoteHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[0].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[0].haloWidth = cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].haloHeight = gpuDim[1];

        remoteHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[1].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[1].haloWidth = cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].haloHeight = gpuDim[1];

        remoteHaloSpecsCpuForGpu[2].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[2].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[2].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1]-cpuForGpuHaloWidth[2];
        remoteHaloSpecsCpuForGpu[2].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[2].haloHeight = cpuForGpuHaloWidth[2];

        remoteHaloSpecsCpuForGpu[3].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[3].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[3].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[3].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[3].haloHeight = cpuForGpuHaloWidth[3];

        localHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[0].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[0].haloWidth = gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];

        localHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[1].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0];
        localHaloSpecsCpuForGpu[1].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[1].haloWidth = gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[1].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];

        localHaloSpecsCpuForGpu[2].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[2].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[2].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1];
        localHaloSpecsCpuForGpu[2].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[2].haloHeight = gpuHaloWidth[2];

        localHaloSpecsCpuForGpu[3].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[3].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[3].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[3].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[3].haloHeight = gpuHaloWidth[3];

        tausch->setLocalHaloInfo2D_CwG(4, localHaloSpecsCpuForGpu);
        tausch->setRemoteHaloInfo2D_CwG(4, remoteHaloSpecsCpuForGpu);

        remoteHaloSpecsGpu = new TauschHaloSpec[4];
        localHaloSpecsGpu = new TauschHaloSpec[4];

        localHaloSpecsGpu[0].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[0].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[0].haloX = gpuHaloWidth[0];
        localHaloSpecsGpu[0].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[0].haloWidth = cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[0].haloHeight = gpuDim[1];

        localHaloSpecsGpu[1].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[2].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[1].haloX = gpuDim[0]+gpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[1].haloWidth = cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].haloHeight = gpuDim[1];

        localHaloSpecsGpu[2].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[1].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[2].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[2].haloY = gpuDim[1]+gpuHaloWidth[3]-cpuForGpuHaloWidth[2];
        localHaloSpecsGpu[2].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[2].haloHeight = cpuForGpuHaloWidth[2];

        localHaloSpecsGpu[3].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[3].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[3].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[3].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[3].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[3].haloHeight = cpuForGpuHaloWidth[3];

        remoteHaloSpecsGpu[0].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[0].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[0].haloX = 0;
        remoteHaloSpecsGpu[0].haloY = 0;
        remoteHaloSpecsGpu[0].haloWidth = gpuHaloWidth[0];
        remoteHaloSpecsGpu[0].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];

        remoteHaloSpecsGpu[1].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[2].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[1].haloX = gpuDim[0]+gpuHaloWidth[0];
        remoteHaloSpecsGpu[1].haloY = 0;
        remoteHaloSpecsGpu[1].haloWidth = gpuHaloWidth[1];
        remoteHaloSpecsGpu[1].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];

        remoteHaloSpecsGpu[2].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[1].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[2].haloX = 0;
        remoteHaloSpecsGpu[2].haloY = gpuDim[1]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[2].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[2].haloHeight = gpuHaloWidth[2];

        remoteHaloSpecsGpu[3].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[3].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[3].haloX = 0;
        remoteHaloSpecsGpu[3].haloY = 0;
        remoteHaloSpecsGpu[3].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[3].haloHeight = gpuHaloWidth[3];

        tausch->setLocalHaloInfo2D_GwC(4, localHaloSpecsGpu);
        tausch->setRemoteHaloInfo2D_GwC(4, remoteHaloSpecsGpu);

    }

    if(gpuonly) {

        size_t tauschGpuDim[2] = {gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1], gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]};
        tausch->enableOpenCL2D(true, 64, true, buildlog);

        gpudat = new double*[numBuffers];
        for(size_t b = 0; b < numBuffers; ++b)
            gpudat[b] = new double[valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3])]{};

        for(size_t j = 0; j < gpuDim[1]; ++j)
            for(size_t i = 0; i < gpuDim[0]; ++i) {
                for(size_t b = 0; b < numBuffers; ++b)
                    for(size_t val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                        gpudat[b][valuesPerPointPerBuffer[b]*((j+gpuHaloWidth[3])*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i+gpuHaloWidth[0])+val] =
                                (b*5 + j*gpuDim[0]+i+1)*10+val;
            }

        try {
            cl_gpudat = new cl::Buffer[numBuffers];
            for(size_t b = 0; b < numBuffers; ++b) {
                size_t s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
                cl_gpudat[b] = cl::Buffer(tausch->getOpenCLContext2D(), &gpudat[b][0], &gpudat[b][s], false);
            }
        } catch(cl::Error error) {
            std::cerr << "Samplecode2D :: constructor :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

        remoteHaloSpecsGpuWithGpu = new TauschHaloSpec[4];
        localHaloSpecsGpuWithGpu = new TauschHaloSpec[4];

        localHaloSpecsGpuWithGpu[0].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpuWithGpu[0].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpuWithGpu[0].haloX = gpuHaloWidth[0]; localHaloSpecsGpuWithGpu[0].haloY = 0;
        localHaloSpecsGpuWithGpu[0].haloWidth = gpuHaloWidth[1]; localHaloSpecsGpuWithGpu[0].haloHeight = gpuHaloWidth[3]+gpuDim[1]+gpuHaloWidth[2];
        localHaloSpecsGpuWithGpu[0].remoteMpiRank = left;
        remoteHaloSpecsGpuWithGpu[0].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpuWithGpu[0].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpuWithGpu[0].haloX = 0; remoteHaloSpecsGpuWithGpu[0].haloY = 0;
        remoteHaloSpecsGpuWithGpu[0].haloWidth = gpuHaloWidth[0]; remoteHaloSpecsGpuWithGpu[0].haloHeight = gpuHaloWidth[3]+gpuDim[1]+gpuHaloWidth[2];
        remoteHaloSpecsGpuWithGpu[0].remoteMpiRank = left;

        localHaloSpecsGpuWithGpu[1].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpuWithGpu[1].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpuWithGpu[1].haloX = gpuDim[0]; localHaloSpecsGpuWithGpu[1].haloY = 0;
        localHaloSpecsGpuWithGpu[1].haloWidth = gpuHaloWidth[0]; localHaloSpecsGpuWithGpu[1].haloHeight = gpuHaloWidth[3]+gpuDim[1]+gpuHaloWidth[2];
        localHaloSpecsGpuWithGpu[1].remoteMpiRank = right;
        remoteHaloSpecsGpuWithGpu[1].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpuWithGpu[1].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpuWithGpu[1].haloX = gpuHaloWidth[0]+gpuDim[0]; remoteHaloSpecsGpuWithGpu[1].haloY = 0;
        remoteHaloSpecsGpuWithGpu[1].haloWidth = gpuHaloWidth[1]; remoteHaloSpecsGpuWithGpu[1].haloHeight = gpuHaloWidth[3]+gpuDim[1]+gpuHaloWidth[2];
        remoteHaloSpecsGpuWithGpu[1].remoteMpiRank = right;

        localHaloSpecsGpuWithGpu[2].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpuWithGpu[2].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpuWithGpu[2].haloX = 0; localHaloSpecsGpuWithGpu[2].haloY = gpuDim[1];
        localHaloSpecsGpuWithGpu[2].haloWidth = gpuHaloWidth[0]+gpuDim[0]+gpuHaloWidth[1]; localHaloSpecsGpuWithGpu[2].haloHeight = gpuHaloWidth[3];
        localHaloSpecsGpuWithGpu[2].remoteMpiRank = top;
        remoteHaloSpecsGpuWithGpu[2].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpuWithGpu[2].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpuWithGpu[2].haloX = 0; remoteHaloSpecsGpuWithGpu[2].haloY = gpuHaloWidth[3]+gpuDim[1];
        remoteHaloSpecsGpuWithGpu[2].haloWidth = gpuHaloWidth[0]+gpuDim[0]+gpuHaloWidth[1]; remoteHaloSpecsGpuWithGpu[2].haloHeight = gpuHaloWidth[2];
        remoteHaloSpecsGpuWithGpu[2].remoteMpiRank = top;

        localHaloSpecsGpuWithGpu[3].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpuWithGpu[3].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpuWithGpu[3].haloX = 0; localHaloSpecsGpuWithGpu[3].haloY = gpuHaloWidth[3];
        localHaloSpecsGpuWithGpu[3].haloWidth = gpuHaloWidth[0]+gpuDim[0]+gpuHaloWidth[1]; localHaloSpecsGpuWithGpu[3].haloHeight = gpuHaloWidth[2];
        localHaloSpecsGpuWithGpu[3].remoteMpiRank = bottom;
        remoteHaloSpecsGpuWithGpu[3].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpuWithGpu[3].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpuWithGpu[3].haloX = 0; remoteHaloSpecsGpuWithGpu[3].haloY = 0;
        remoteHaloSpecsGpuWithGpu[3].haloWidth = gpuHaloWidth[0]+gpuDim[0]+gpuHaloWidth[1]; remoteHaloSpecsGpuWithGpu[3].haloHeight = gpuHaloWidth[3];
        remoteHaloSpecsGpuWithGpu[3].remoteMpiRank = bottom;

        tausch->setLocalHaloInfo2D_GwG(4, localHaloSpecsGpuWithGpu);
        tausch->setRemoteHaloInfo2D_GwG(4, remoteHaloSpecsGpuWithGpu);


    }

#endif

}

Sample::~Sample() {

    delete[] valuesPerPointPerBuffer;
    delete tausch;

#ifdef OPENCL
    if(!gpuonly) {
#endif

        for(size_t b = 0; b < numBuffers; ++b)
            delete[] dat[b];
        delete[] dat;

#ifdef OPENCL
    }


    if(hybrid) {
        delete[] localHaloSpecsGpu;
        delete[] remoteHaloSpecsGpu;
        delete[] localHaloSpecsCpuForGpu;
        delete[] remoteHaloSpecsCpuForGpu;
    }

    if(gpuonly) {
        delete[] localHaloSpecsGpuWithGpu;
        delete[] remoteHaloSpecsGpuWithGpu;
    }

    if(hybrid || gpuonly) {
        for(size_t b = 0; b < numBuffers; ++b)
            delete[] gpudat[b];
        delete[] gpudat;
        delete[] cl_gpudat;
    }

#endif

}

void Sample::launchCPU() {

#ifdef OPENCL
    if(hybrid) {

        for(size_t iter = 0; iter < loops; ++iter) {

            int sendtagsCpu[4] = {0, 1, 2, 3};
            int recvtagsCpu[4] = {1, 0, 3, 2};
            int sendtagsGpu[4] = {0, 1, 2, 3};
            int recvtagsGpu[4] = {0, 1, 2, 3};

            tausch->postAllReceives2D_CwC(recvtagsCpu);
            tausch->postAllReceives2D_GwC(recvtagsGpu);

            for(size_t ver_hor = 0; ver_hor < 2; ++ver_hor) {

                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer2D_CwC(2*ver_hor, b, dat[b]);
                tausch->send2D_CwC(2*ver_hor, sendtagsCpu[2*ver_hor]);

                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer2D_CwC(2*ver_hor+1, b, dat[b]);
                tausch->send2D_CwC(2*ver_hor+1, sendtagsCpu[2*ver_hor +1]);

                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer2D_CwG(2*ver_hor, b, dat[b]);
                tausch->send2D_CwG(2*ver_hor, sendtagsGpu[2*ver_hor]);

                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer2D_CwG(2*ver_hor+1, b, dat[b]);
                tausch->send2D_CwG(2*ver_hor+1, sendtagsGpu[2*ver_hor +1]);

                tausch->recv2D_CwC(2*ver_hor);
                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer2D_CwC(2*ver_hor, b, dat[b]);

                tausch->recv2D_CwC(2*ver_hor+1);
                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer2D_CwC(2*ver_hor+1, b, dat[b]);

                tausch->recv2D_CwG(2*ver_hor);
                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer2D_CwG(2*ver_hor, b, dat[b]);

                tausch->recv2D_CwG(2*ver_hor+1);
                for(size_t b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer2D_CwG(2*ver_hor+1, b, dat[b]);

            }

        }

    } else {

#endif

        for(size_t iter = 0; iter < loops; ++iter) {

            int sendtags[4] = {0, 1, 2, 3};
            int recvtags[4] = {1, 0, 3, 2};

            tausch->postAllReceives2D_CwC(recvtags);

//            tausch->postReceive2D_CwC(allRemoteHaloIds[0], recvtags[0]);
//            tausch->postReceive2D_CwC(allRemoteHaloIds[1], recvtags[1]);
//            tausch->postReceive2D_CwC(allRemoteHaloIds[2], recvtags[2]);
//            tausch->postReceive2D_CwC(allRemoteHaloIds[3], recvtags[3]);

            for(size_t ver_hor = 0; ver_hor < 2; ++ver_hor) {

                tausch->packAndSend2D_CwC(allLocalHaloIds[2*ver_hor], dat[0], sendtags[2*ver_hor]);
//                for(size_t b = 0; b < numBuffers; ++b)
//                    tausch->packSendBuffer2D_CwC(allLocalHaloIds[2*ver_hor], b, dat[b]);
//                tausch->send2D_CwC(allLocalHaloIds[2*ver_hor], sendtags[2*ver_hor]);

                tausch->packAndSend2D_CwC(allLocalHaloIds[2*ver_hor +1], dat[0], sendtags[2*ver_hor +1]);
//                for(size_t b = 0; b < numBuffers; ++b)
//                    tausch->packSendBuffer2D_CwC(allLocalHaloIds[2*ver_hor+1], b, dat[b]);
//                tausch->send2D_CwC(allLocalHaloIds[2*ver_hor+1], sendtags[2*ver_hor +1]);

                tausch->recvAndUnpack2D_CwC(allRemoteHaloIds[2*ver_hor], dat[0]);
//                tausch->recv2D_CwC(allRemoteHaloIds[2*ver_hor]);
//                for(size_t b = 0; b < numBuffers; ++b)
//                    tausch->unpackRecvBuffer2D_CwC(allRemoteHaloIds[2*ver_hor], b, dat[b]);

                tausch->recvAndUnpack2D_CwC(allRemoteHaloIds[2*ver_hor +1], dat[0]);
//                tausch->recv2D_CwC(allRemoteHaloIds[2*ver_hor+1]);
//                for(size_t b = 0; b < numBuffers; ++b)
//                    tausch->unpackRecvBuffer2D_CwC(allRemoteHaloIds[2*ver_hor+1], b, dat[b]);

            }

        }

#ifdef OPENCL
    }
#endif

}

#ifdef OPENCL
void Sample::launchGPU() {

    for(size_t iter = 0; iter < loops; ++iter) {

        int sendtags[4] = {0, 1, 2, 3};
        int recvtags[4] = {0, 1, 2, 3};

        tausch->postAllReceives2D_GwC(recvtags);

        for(size_t ver_hor = 0; ver_hor < 2; ++ver_hor) {

            tausch->recv2D_GwC(2*ver_hor);
            for(size_t b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer2D_GwC(2*ver_hor, b, cl_gpudat[b]);

            tausch->recv2D_GwC(2*ver_hor+1);
            for(size_t b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer2D_GwC(2*ver_hor+1, b, cl_gpudat[b]);

            for(size_t b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer2D_GwC(2*ver_hor, b, cl_gpudat[b]);
            tausch->send2D_GwC(2*ver_hor, sendtags[2*ver_hor]);

            for(size_t b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer2D_GwC(2*ver_hor+1, b, cl_gpudat[b]);
            tausch->send2D_GwC(2*ver_hor+1, sendtags[2*ver_hor +1]);

        }

    }

    for(size_t b = 0; b < numBuffers; ++b) {
        size_t s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
        cl::copy(tausch->getOpenCLQueue2D(), cl_gpudat[b], &gpudat[b][0], &gpudat[b][s]);
    }

}

void Sample::launchGPUonly() {

    for(size_t iter = 0; iter < loops; ++iter) {

        int sendtags[4] = {0, 1, 2, 3};
        int recvtags[4] = {1, 0, 3, 2};

        tausch->postAllReceives2D_GwG(recvtags);

        for(size_t ver_hor = 0; ver_hor < 2; ++ver_hor) {

            for(size_t b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer2D_GwG(2*ver_hor, b, cl_gpudat[b]);
            tausch->send2D_GwG(2*ver_hor, sendtags[2*ver_hor]);

            for(size_t b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer2D_GwG(2*ver_hor+1, b, cl_gpudat[b]);
            tausch->send2D_GwG(2*ver_hor+1, sendtags[2*ver_hor +1]);

            tausch->recv2D_GwG(2*ver_hor);
            for(size_t b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer2D_GwG(2*ver_hor, b, cl_gpudat[b]);

            tausch->recv2D_GwG(2*ver_hor+1);
            for(size_t b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer2D_GwG(2*ver_hor+1, b, cl_gpudat[b]);

        }

    }

    for(size_t b = 0; b < numBuffers; ++b) {
        size_t s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
        cl::copy(tausch->getOpenCLQueue2D(), cl_gpudat[b], &gpudat[b][0], &gpudat[b][s]);
    }

}
#endif

void Sample::printCPU() {

    for(int j = localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {
        for(size_t b = 0; b < numBuffers; ++b) {
            for(size_t val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
                for(size_t i = 0; i < localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << dat[b][valuesPerPointPerBuffer[b]*(j*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i) + val] << " ";
                if(val != valuesPerPointPerBuffer[b]-1)
                    std::cout << "   ";
            }
            if(b != numBuffers-1)
                std::cout << "          ";
        }
        std::cout << std::endl;
    }

}

#ifdef OPENCL
void Sample::printGPU() {

    for(int j = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]-1; j >= 0; --j) {
        for(size_t b = 0; b < numBuffers; ++b) {
            for(size_t val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
                for(size_t i = 0; i < gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << gpudat[b][valuesPerPointPerBuffer[b]*(j*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i) + val] << " ";
                if(val != valuesPerPointPerBuffer[b]-1)
                    std::cout << "   ";
            }
            if(b != numBuffers-1)
                std::cout << "          ";
        }
        std::cout << std::endl;
    }

}
#endif
