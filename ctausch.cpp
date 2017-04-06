#include "tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch.h"

    CTausch *tausch_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int haloWidth, MPI_Comm comm) {
        Tausch *t = new Tausch(localDimX, localDimY, mpiNumX, mpiNumY, haloWidth, comm);
        return reinterpret_cast<CTausch*>(t);
    }

    void tausch_delete(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        delete t;
    }

    void tausch_getMPICommunicator(CTausch *tC, MPI_Comm *comm) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        *comm = t->getMPICommunicator();
    }

    void tausch_postCpuReceives(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->postCpuReceives();
    }

    void tausch_performCpuToCpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performCpuToCpu();
    }

    void tausch_startCpuEdge(CTausch *tC, enum Edge edge) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->startCpuEdge(edge==TauschLeft ? Tausch::Left : (edge==TauschRight ? Tausch::Right : (edge==TauschTop ? Tausch::Top : Tausch::Bottom)));
    }

    void tausch_completeCpuEdge(CTausch *tC, enum Edge edge) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->completeCpuEdge(edge==TauschLeft ? Tausch::Left : (edge==TauschRight ? Tausch::Right : (edge==TauschTop ? Tausch::Top : Tausch::Bottom)));
    }

    void tausch_setCPUData(CTausch *tC, real_t *dat) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->setCPUData(dat);
    }

#ifdef TAUSCH_OPENCL
    void tausch_enableOpenCL(CTausch *tC, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->enableOpenCL(blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
    }

    void tausch_setOpenCLInfo(CTausch *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, bool blockingSyncCpuGpu) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        cl::Device *dev = new cl::Device(*clDefaultDevice);
        cl::Context *con = new cl::Context(*clContext);
        cl::CommandQueue *que = new cl::CommandQueue(*clQueue);
        t->enableOpenCL(*dev, *con, *que, blockingSyncCpuGpu);
    }

    void tausch_setGPUData(CTausch *tC, cl_mem dat, int gpuDimX, int gpuDimY) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        cl::Buffer *buf = new cl::Buffer(dat);
        t->setGPUData(*buf, gpuDimX, gpuDimY);
    }

    void tausch_performCpuToCpuAndCpuToGpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performCpuToCpuAndCpuToGpu();
    }

    void tausch_performCpuToGpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performCpuToGpu();
    }

    void tausch_performGpuToCpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performGpuToCpu();
    }

    void tausch_startCpuToGpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->startCpuToGpu();
    }
    void tausch_startGpuToCpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->startGpuToCpu();
    }

    void tausch_completeCpuToGpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->completeCpuToGpu();
    }
    void tausch_completeGpuToCpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->completeGpuToCpu();
    }

    cl_context tausch_getContext(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        return t->getContext()();
    }

    cl_command_queue tausch_getQueue(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        return t->getQueue()();
    }
#endif

#ifdef __cplusplus
}
#endif
