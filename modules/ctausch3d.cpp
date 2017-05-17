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

    void tausch3d_performCpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpuData();
    }

    void tausch3d_startCpuDataEdge(CTausch3D *tC, enum Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuDataEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
                                            (edge==TAUSCH_RIGHT ? Tausch3D::RIGHT :
                                                                  (edge==TAUSCH_TOP ? Tausch3D::TOP :
                                                                                      (edge==TAUSCH_BOTTOM ? Tausch3D::BOTTOM :
                                                                                                             edge==TAUSCH_FRONT ? Tausch3D::FRONT :
                                                                                                                                  Tausch3D::BACK))));
    }

    void tausch3d_completeCpuDataEdge(CTausch3D *tC, enum Edge edge) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuDataEdge(edge==TAUSCH_LEFT ? Tausch3D::LEFT :
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

#ifdef TAUSCH_OPENCL
    void tausch3d_enableOpenCL(CTausch3D *tC, int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->enableOpenCL(gpuHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
    }

    void tausch3d_setOpenCLInfo(CTausch3D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue,
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

    void tausch3d_performCpuToCpuDataAndCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToCpuDataAndCpuToGpuData();
    }

    void tausch3d_performCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performCpuToGpuData();
    }

    void tausch3d_performGpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->performGpuToCpuData();
    }

    void tausch3d_startCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startCpuToGpuData();
    }
    void tausch3d_startGpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->startGpuToCpuData();
    }

    void tausch3d_completeCpuToGpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeCpuToGpuData();
    }
    void tausch3d_completeGpuToCpuData(CTausch3D *tC) {
        Tausch3D *t = reinterpret_cast<Tausch3D*>(tC);
        t->completeGpuToCpuData();
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
