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

    void tausch3d_postCpuDataReceives(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->postCpuDataReceives();
    }

    void tausch3d_postCpuStencilReceives(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->postCpuStencilReceives();
    }

    void tausch3d_performCpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpuData();
    }

    void tausch3d_performCpuToCpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpuStencil();
    }

    void tausch3d_startCpuDataEdge(CTausch3D *tC, Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuDataEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
                                             (edge==TAUSCH_RIGHT ? Tausch3D::RIGHT :
                                                                (edge==TAUSCH_TOP ? Tausch3D::TOP :
                                                                                 (edge==TAUSCH_BOTTOM ? Tausch3D::BOTTOM :
                                                                                                      edge==TAUSCH_FRONT ? Tausch3D::FRONT :
                                                                                                                           Tausch3D::BACK))));
    }

    void tausch3d_startCpuStencilEdge(CTausch3D *tC, Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuStencilEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
                                                (edge==TAUSCH_RIGHT ? Tausch3D::RIGHT :
                                                                   (edge==TAUSCH_TOP ? Tausch3D::TOP :
                                                                                    (edge==TAUSCH_BOTTOM ? Tausch3D::BOTTOM :
                                                                                                         edge==TAUSCH_FRONT ? Tausch3D::FRONT :
                                                                                                                              Tausch3D::BACK))));
    }

    void tausch3d_completeCpuDataEdge(CTausch3D *tC, Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuDataEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
                                                 (edge==TAUSCH_RIGHT ? Tausch3D::RIGHT :
                                                                     (edge==TAUSCH_TOP ? Tausch3D::TOP :
                                                                                       (edge==TAUSCH_BOTTOM ? Tausch3D::BOTTOM :
                                                                                                            edge==TAUSCH_FRONT ? Tausch3D::FRONT :
                                                                                                                                 Tausch3D::BACK))));
    }

    void tausch3d_completeCpuStencilEdge(CTausch3D *tC, Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuStencilEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
                                                   (edge==TAUSCH_RIGHT ? Tausch3D::RIGHT :
                                                                      (edge==TAUSCH_TOP ? Tausch3D::TOP :
                                                                                       (edge==TAUSCH_BOTTOM ? Tausch3D::BOTTOM :
                                                                                                            edge==TAUSCH_FRONT ? Tausch3D::FRONT :
                                                                                                                                 Tausch3D::BACK))));
    }

    void tausch3d_setCpuData(CTausch3D *tC, real_t *dat) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->setCpuData(dat);
    }

    void tausch3d_setCpuStencil(CTausch3D *tC, real_t *stencil, int stencilNumPoints) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->setCpuStencil(stencil, stencilNumPoints);
    }

#ifdef TAUSCH_OPENCL
    void tausch3d_enableOpenCL(CTausch3D *tC, int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->enableOpenCL(gpuHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
    }

    void tausch3d_enableOpenCLWithInfo(CTausch3D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue,
                                       int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        cl::Device *dev = new cl::Device(*clDefaultDevice);
        cl::Context *con = new cl::Context(*clContext);
        cl::CommandQueue *que = new cl::CommandQueue(*clQueue);
        t->enableOpenCL(*dev, *con, *que, gpuHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize);
    }

    void tausch3d_setGpuData(CTausch3D *tC, cl_mem dat, int *gpuDim) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        cl::Buffer *buf = new cl::Buffer(dat);
        t->setGpuData(*buf, gpuDim);
    }

    void tausch3d_setGpuStencil(CTausch3D *tC, cl_mem stencil, int *gpuDim, int stencilNumPoints) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        cl::Buffer *buf = new cl::Buffer(stencil);
        t->setGpuStencil(*buf, stencilNumPoints, gpuDim);
    }

    void tausch3d_performCpuToCpuDataAndCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpuDataAndCpuToGpuData();
    }

    void tausch3d_performCpuToCpuStencilAndCpuToGpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpuStencilAndCpuToGpuStencil();
    }

    void tausch3d_performCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToGpuData();
    }

    void tausch3d_performCpuToGpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToGpuStencil();
    }

    void tausch3d_performGpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performGpuToCpuData();
    }

    void tausch3d_performGpuToCpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performGpuToCpuStencil();
    }

    void tausch3d_startCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuToGpuData();
    }

    void tausch3d_startCpuToGpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuToGpuStencil();
    }

    void tausch3d_startGpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startGpuToCpuData();
    }

    void tausch3d_startGpuToCpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startGpuToCpuStencil();
    }

    void tausch3d_completeCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuToGpuData();
    }

    void tausch3d_completeCpuToGpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuToGpuStencil();
    }

    void tausch3d_completeGpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeGpuToCpuData();
    }

    void tausch3d_completeGpuToCpuStencil(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeGpuToCpuStencil();
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
