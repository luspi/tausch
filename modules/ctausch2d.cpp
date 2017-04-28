#include "tausch2d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch2d.h"

    CTausch2D *tausch2d_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int cpuHaloWidth, MPI_Comm comm) {
        Tausch2D *t = new Tausch2D(localDimX, localDimY, mpiNumX, mpiNumY, cpuHaloWidth, comm);
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

    void tausch2d_postCpuDataReceives(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->postCpuDataReceives();
    }
    void tausch2d_postCpuStencilReceives(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->postCpuStencilReceives();
    }

    void tausch2d_performCpuToCpuData(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->performCpuToCpuData();
    }
    void tausch2d_performCpuToCpuStencil(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->performCpuToCpuStencil();
    }

    void tausch2d_startCpuDataEdge(CTausch2D *tC, enum Edge edge) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->startCpuDataEdge(edge==TAUSCH_LEFT ? Tausch2D::LEFT : (edge==TAUSCH_RIGHT ? Tausch2D::RIGHT : (edge==TAUSCH_TOP ? Tausch2D::TOP : Tausch2D::BOTTOM)));
    }
    void tausch2d_startCpuStencilEdge(CTausch2D *tC, enum Edge edge) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->startCpuStencilEdge(edge==TAUSCH_LEFT ? Tausch2D::LEFT : (edge==TAUSCH_RIGHT ? Tausch2D::RIGHT : (edge==TAUSCH_TOP ? Tausch2D::TOP : Tausch2D::BOTTOM)));
    }

    void tausch2d_completeCpuDataEdge(CTausch2D *tC, enum Edge edge) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->completeCpuDataEdge(edge==TAUSCH_LEFT ? Tausch2D::LEFT : (edge==TAUSCH_RIGHT ? Tausch2D::RIGHT : (edge==TAUSCH_TOP ? Tausch2D::TOP : Tausch2D::BOTTOM)));
    }
    void tausch2d_completeCpuStencilEdge(CTausch2D *tC, enum Edge edge) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->completeCpuStencilEdge(edge==TAUSCH_LEFT ? Tausch2D::LEFT : (edge==TAUSCH_RIGHT ? Tausch2D::RIGHT : (edge==TAUSCH_TOP ? Tausch2D::TOP : Tausch2D::BOTTOM)));
    }

    void tausch2d_setCPUData(CTausch2D *tC, real_t *dat) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->setCPUData(dat);
    }
    void tausch2d_setCPUStencil(CTausch2D *tC, real_t *dat, int stencilNumPoints) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->setCPUStencil(dat, stencilNumPoints);
    }

#ifdef TAUSCH_OPENCL
    void tausch2d_enableOpenCL(CTausch2D *tC, int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->enableOpenCL(gpuHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
    }

    void tausch2d_setOpenCLInfo(CTausch2D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, int gpuHaloWidth, bool blockingSyncCpuGpu) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        cl::Device *dev = new cl::Device(*clDefaultDevice);
        cl::Context *con = new cl::Context(*clContext);
        cl::CommandQueue *que = new cl::CommandQueue(*clQueue);
        t->enableOpenCL(*dev, *con, *que, gpuHaloWidth, blockingSyncCpuGpu);
    }

    void tausch2d_setGPUData(CTausch2D *tC, cl_mem dat, int gpuDimX, int gpuDimY) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        cl::Buffer *buf = new cl::Buffer(dat);
        t->setGPUData(*buf, gpuDimX, gpuDimY);
    }

    void tausch2d_performCpuToCpuDataAndCpuToGpuData(CTausch2D *tC) {
        Tausch2D *t = reinterpret_cast<Tausch2D*>(tC);
        t->performCpuToCpuDataAndCpuToGpuData();
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
