#include "tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch.h"

    CTausch *tausch_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool blockingSyncCpuGpu, bool withOpenCL, bool setupOpenCL, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch *t = new Tausch(localDimX, localDimY, mpiNumX, mpiNumY, blockingSyncCpuGpu, withOpenCL, setupOpenCL, clLocalWorkgroupSize, giveOpenCLDeviceName);
        return reinterpret_cast<CTausch*>(t);
    }

    CTausch *tausch_newCpuAndGpu(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool blockingSyncCpuGpu, bool withOpenCL, bool setupOpenCL, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
        Tausch *t = new Tausch(localDimX, localDimY, mpiNumX, mpiNumY, blockingSyncCpuGpu, withOpenCL, setupOpenCL, clLocalWorkgroupSize, giveOpenCLDeviceName);
        return reinterpret_cast<CTausch*>(t);
    }

    CTausch *tausch_newCpu(int localDimX, int localDimY, int mpiNumX, int mpiNumY) {
        Tausch *t = new Tausch(localDimX, localDimY, mpiNumX, mpiNumY);
        return reinterpret_cast<CTausch*>(t);
    }

    void tausch_delete(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        delete t;
    }

    void tausch_setOpenCLInfo(CTausch *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        cl::Device *dev = new cl::Device(*clDefaultDevice);
        cl::Context *con = new cl::Context(*clContext);
        cl::CommandQueue *que = new cl::CommandQueue(*clQueue);
        t->setOpenCLInfo(*dev, *con, *que);
    }

    void tausch_postCpuReceives(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->postCpuReceives();
    }

    void tausch_performCpuToCpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performCpuToCpuTausch();
    }

    void tausch_performCpuToCpuAndCpuToGpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performCpuToCpuAndCpuToGpuTausch();
    }

    void tausch_performCpuToGpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performCpuToGpuTausch();
    }

    void tausch_performGpuToCpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->performGpuToCpuTausch();
    }

    void tausch_startCpuToGpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->startCpuToGpuTausch();
    }
    void tausch_startGpuToCpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->startGpuToCpuTausch();
    }

    void tausch_completeCpuToGpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->completeCpuToGpuTausch();
    }
    void tausch_completeGpuToCpuTausch(CTausch *tC) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->completeGpuToCpuTausch();
    }

    void tausch_startCpuTauschEdge(CTausch *tC, Edge edge) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->startCpuTauschEdge(edge);
    }

    void tausch_completeCpuTauschEdge(CTausch *tC, Edge edge) {
        Tausch *t = reinterpret_cast<Tausch*>(tC);
        t->completeCpuTauschEdge(edge);
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
