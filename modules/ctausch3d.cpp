#include "tausch3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch3d.h"

    CTausch3D *tausch3d_new(int *localDim, int *mpiNum, int *haloWidth, MPI_Comm comm) {
        Tausch3D *t = new Tausch3D(localDim, mpiNum, haloWidth, comm);
        return reinterpret_cast<CTausch3D*>(t);
    }

    void tausch3d_delete(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        delete t;
    }

    void tausch3d_getMPICommunicator(CTausch3D *tC, MPI_Comm *comm) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        *comm = t->getMPICommunicator();
    }

    void tausch3d_postCpuReceives(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->postCpuReceives();
    }

    void tausch3d_performCpuToCpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpu();
    }

    void tausch3d_startCpuEdge(CTausch3D *tC, enum Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
                                            (edge==TAUSCH_RIGHT ? Tausch3D::RIGHT :
                                                                  (edge==TAUSCH_TOP ? Tausch3D::TOP :
                                                                                      (edge==TAUSCH_BOTTOM ? Tausch3D::BOTTOM :
                                                                                                             edge==TAUSCH_FRONT ? Tausch3D::FRONT :
                                                                                                                                  Tausch3D::BACK))));
    }

    void tausch3d_completeCpuEdge(CTausch3D *tC, enum Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
                                               (edge==TAUSCH_RIGHT ? Tausch3D::RIGHT :
                                                                     (edge==TAUSCH_TOP ? Tausch3D::TOP :
                                                                                         (edge==TAUSCH_BOTTOM ? Tausch3D::BOTTOM :
                                                                                                                edge==TAUSCH_FRONT ? Tausch3D::FRONT :
                                                                                                                                     Tausch3D::BACK))));
    }

    void tausch3d_setCPUData(CTausch3D *tC, real_t *dat) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->setCPUData(dat);
    }

#ifdef TAUSCH_OPENCL
    void tausch3d_enableOpenCL(CTausch3D *tC, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->enableOpenCL(blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
    }

    void tausch3d_setOpenCLInfo(CTausch3D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, bool blockingSyncCpuGpu) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        cl::Device *dev = new cl::Device(*clDefaultDevice);
        cl::Context *con = new cl::Context(*clContext);
        cl::CommandQueue *que = new cl::CommandQueue(*clQueue);
        t->enableOpenCL(*dev, *con, *que, blockingSyncCpuGpu);
    }

    void tausch3d_setGPUData(CTausch3D *tC, cl_mem dat, int *gpuDim) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        cl::Buffer *buf = new cl::Buffer(dat);
        t->setGPUData(*buf, gpuDim);
    }

    void tausch3d_performCpuToCpuAndCpuToGpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpuAndCpuToGpu();
    }

    void tausch3d_performCpuToGpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToGpu();
    }

    void tausch3d_performGpuToCpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performGpuToCpu();
    }

    void tausch3d_startCpuToGpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuToGpu();
    }
    void tausch3d_startGpuToCpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startGpuToCpu();
    }

    void tausch3d_completeCpuToGpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuToGpu();
    }
    void tausch3d_completeGpuToCpu(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeGpuToCpu();
    }

    cl_context tausch3d_getContext(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        return t->getContext()();
    }

    cl_command_queue tausch3d_getQueue(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        return t->getQueue()();
    }
#endif

#ifdef __cplusplus
}
#endif
