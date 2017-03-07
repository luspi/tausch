#include "tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch.h"

    CTausch *tausch_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY) {
        Tausch *t = new Tausch(localDimX, localDimY, mpiNumX, mpiNumY);
        return reinterpret_cast<CTausch*>(t);
    }

    void tausch_delete(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        delete t;
    }

    void tausch_enableOpenCL(CTausch *tC, bool blockingSyncCpuGpu, bool setupOpenCL, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->enableOpenCL(blockingSyncCpuGpu, setupOpenCL, clLocalWorkgroupSize, giveOpenCLDeviceName);
    }

    void tausch_setOpenCLInfo(CTausch *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, bool blockingSyncCpuGpu) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        cl::Device *dev = new cl::Device(*clDefaultDevice);
        cl::Context *con = new cl::Context(*clContext);
        cl::CommandQueue *que = new cl::CommandQueue(*clQueue);
        t->enableOpenCL(*dev, *con, *que, blockingSyncCpuGpu);
    }

    void tausch_postCpuReceives(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->postCpuReceives();
    }

    void tausch_performCpuToCpu(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performCpuToCpu();
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

    void tausch_startCpuEdge(CTausch *tC, Edge edge) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->startCpuEdge(edge);
    }

    void tausch_completeCpuEdge(CTausch *tC, Edge edge) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->completeCpuEdge(edge);
    }

    void tausch_syncCpuAndGpu(CTausch *tC, bool iAmTheCPU) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->syncCpuAndGpu(iAmTheCPU);
    }


    void tausch_setHaloWidth(CTausch *tC, int haloWidth) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->setHaloWidth(haloWidth);
    }
    void tausch_setCPUData(CTausch *tC, double *dat) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->setCPUData(dat);
    }
    void tausch_setGPUData(CTausch *tC, cl_mem dat, int gpuDimX, int gpuDimY) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        cl::Buffer *buf = new cl::Buffer(dat);
        t->setGPUData(*buf, gpuDimX, gpuDimY);
    }
    bool tausch_isGpuEnabled(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        return t->isGpuEnabled();
    }

    cl_context tausch_getContext(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        return t->cl_context();
    }

    cl_command_queue tausch_getQueue(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        return t->cl_queue();
    }

    void tausch_checkOpenCLError(CTausch *tC, cl_int clErr, char *loc) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->checkOpenCLError(clErr, std::string(loc));
    }

#ifdef __cplusplus
}
#endif
