#include "tausch2d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch2d.h"

    CTausch2D *tausch2d_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int haloWidth, MPI_Comm comm) {
        Tausch2D *t = new Tausch2D(localDimX, localDimY, mpiNumX, mpiNumY, haloWidth, comm);
        return reinterpret_cast<CTausch2D*>(t);
    }

    void tausch2d_delete(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        delete t;
    }

    void tausch2d_getMPICommunicator(CTausch2D *tC, MPI_Comm *comm) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        *comm = t->getMPICommunicator();
    }

    void tausch2d_postCpuReceives(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->postCpuReceives();
    }

    void tausch2d_performCpuToCpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->performCpuToCpu();
    }

    void tausch2d_startCpuEdge(CTausch2D *tC, enum Edge edge) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->startCpuEdge(edge==TauschLeft ? Tausch2D::Left : (edge==TauschRight ? Tausch2D::Right : (edge==TauschTop ? Tausch2D::Top : Tausch2D::Bottom)));
    }

    void tausch2d_completeCpuEdge(CTausch2D *tC, enum Edge edge) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->completeCpuEdge(edge==TauschLeft ? Tausch2D::Left : (edge==TauschRight ? Tausch2D::Right : (edge==TauschTop ? Tausch2D::Top : Tausch2D::Bottom)));
    }

    void tausch2d_setCPUData(CTausch2D *tC, real_t *dat) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->setCPUData(dat);
    }

#ifdef TAUSCH_OPENCL
    void tausch2d_enableOpenCL(CTausch2D *tC, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->enableOpenCL(blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
    }

    void tausch2d_setOpenCLInfo(CTausch2D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, bool blockingSyncCpuGpu) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        cl::Device *dev = new cl::Device(*clDefaultDevice);
        cl::Context *con = new cl::Context(*clContext);
        cl::CommandQueue *que = new cl::CommandQueue(*clQueue);
        t->enableOpenCL(*dev, *con, *que, blockingSyncCpuGpu);
    }

    void tausch2d_setGPUData(CTausch2D *tC, cl_mem dat, int gpuDimX, int gpuDimY) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        cl::Buffer *buf = new cl::Buffer(dat);
        t->setGPUData(*buf, gpuDimX, gpuDimY);
    }

    void tausch2d_performCpuToCpuAndCpuToGpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->performCpuToCpuAndCpuToGpu();
    }

    void tausch2d_performCpuToGpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->performCpuToGpu();
    }

    void tausch2d_performGpuToCpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->performGpuToCpu();
    }

    void tausch2d_startCpuToGpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->startCpuToGpu();
    }
    void tausch2d_startGpuToCpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->startGpuToCpu();
    }

    void tausch2d_completeCpuToGpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->completeCpuToGpu();
    }
    void tausch2d_completeGpuToCpu(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->completeGpuToCpu();
    }

    cl_context tausch2d_getContext(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        return t->getContext()();
    }

    cl_command_queue tausch2d_getQueue(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        return t->getQueue()();
    }
#endif

#ifdef __cplusplus
}
#endif
