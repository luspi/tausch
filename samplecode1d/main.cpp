#include "sample.h"
#include <cmath>
#include <sstream>
#include <chrono>
#include <future>

int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided);

    // If this feature is not available -> abort
    if(provided != MPI_THREAD_SERIALIZED){
        std::cout << "ERROR: The MPI library does not have full thread support at level MPI_THREAD_SERIALIZED... Abort!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    size_t localDim = 5;
    size_t gpuDim = 3;
    size_t loops = 1;
    int printMpiRank = -1;
    size_t cpuHaloWidth[2] = {1,1};
    size_t gpuHaloWidth[2] = {1,1};
    size_t cpuForGpuHaloWidth[2] = {1,1};
    bool gpu = true;

    if(argc > 1) {
        for(int i = 1; i < argc; ++i) {

            if(argv[i] == std::string("-x") && i < argc-1)
                localDim = atoi(argv[++i]);
            else if(argv[i] == std::string("-gx") && i < argc-1)
                gpuDim = atoi(argv[++i]);
            else if(argv[i] == std::string("-num") && i < argc-1)
                loops = atoi(argv[++i]);
            else if(argv[i] == std::string("-print") && i < argc-1)
                printMpiRank = atoi(argv[++i]);
            else if(argv[i] == std::string("-cpu"))
                gpu = false;
            else if(argv[i] == std::string("-chalo") && i < argc-1) {
                std::string arg(argv[++i]);
                size_t last = 0;
                size_t next = 0;
                int count = 0;
                while((next = arg.find(",", last)) != std::string::npos && count < 1) {
                    std::stringstream str;
                    str << arg.substr(last, next-last);
                    str >> cpuHaloWidth[count];
                    last = next + 1;
                    ++count;
                }
                if(count != 1) {
                    int tmpHalo = 0;
                    std::stringstream str;
                    str << arg;
                    str >> tmpHalo;
                    for(int i = 0; i < 2; ++i)
                        cpuHaloWidth[i] = tmpHalo;
                } else {
                    std::stringstream str;
                    str << arg.substr(last);
                    str >> cpuHaloWidth[1];
                }
            } else if(argv[i] == std::string("-ghalo") && i < argc-1) {
                std::string arg(argv[++i]);
                size_t last = 0;
                size_t next = 0;
                int count = 0;
                while((next = arg.find(",", last)) != std::string::npos && count < 1) {
                    std::stringstream str;
                    str << arg.substr(last, next-last);
                    str >> gpuHaloWidth[count];
                    last = next + 1;
                    ++count;
                }
                if(count != 1) {
                    int tmpHalo = 0;
                    std::stringstream str;
                    str << arg;
                    str >> tmpHalo;
                    for(int i = 0; i < 2; ++i)
                        gpuHaloWidth[i] = tmpHalo;
                } else {
                    std::stringstream str;
                    str << arg.substr(last);
                    str >> gpuHaloWidth[1];
                }
            } else if(argv[i] == std::string("-cghalo") && i < argc-1) {
                std::string arg(argv[++i]);
                size_t last = 0;
                size_t next = 0;
                int count = 0;
                while((next = arg.find(",", last)) != std::string::npos && count < 1) {
                    std::stringstream str;
                    str << arg.substr(last, next-last);
                    str >> cpuForGpuHaloWidth[count];
                    last = next + 1;
                    ++count;
                }
                if(count != 1) {
                    int tmpHalo = 0;
                    std::stringstream str;
                    str << arg;
                    str >> tmpHalo;
                    for(int i = 0; i < 2; ++i)
                        cpuForGpuHaloWidth[i] = tmpHalo;
                } else {
                    std::stringstream str;
                    str << arg.substr(last);
                    str >> cpuForGpuHaloWidth[1];
                }
            }
        }
    }

    if(mpiRank == 0) {

        std::cout << std::endl
                  << "localDim      = " << localDim << std::endl
                  << "gpuDim        = " << gpuDim << std::endl
                  << "loops         = " << loops << std::endl
                  << "CPU halo      = " << cpuHaloWidth[0] << "/" << cpuHaloWidth[1] << std::endl
                  << "GPU halo      = " << gpuHaloWidth[0] << "/" << gpuHaloWidth[1] << std::endl
                  << "CPU->GPU halo = " << cpuForGpuHaloWidth[0] << "/" << cpuForGpuHaloWidth[1] << std::endl
                  << "Version       = " << (gpu ? "Hybrid" : "CPU-only") << std::endl
                  << std::endl;

    }
    Sample sample(localDim, gpuDim, loops, cpuHaloWidth, gpuHaloWidth, cpuForGpuHaloWidth, gpu);

    if(mpiRank == printMpiRank) {
        std::cout << "-------------------------------" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        std::cout << "CPU region BEFORE" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        sample.print();
        std::cout << "-------------------------------" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::steady_clock::now();

    if(gpu) {

        std::future<void> thrdGPU(std::async(std::launch::async, &Sample::launchGPU, &sample));
        sample.launchCPU();
        thrdGPU.wait();

    } else {

        sample.launchCPU();

    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();

    if(mpiRank == printMpiRank) {
        std::cout << "-------------------------------" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        std::cout << "CPU region AFTER" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        sample.print();
        std::cout << "-------------------------------" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(mpiRank == 0)
        std::cout << "Time required: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    MPI_Finalize();

    return 0;

}
