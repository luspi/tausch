#include "sample.h"
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

    int localDimX = 25, localDimY = 15;
    double portionGPU = 0.5;
    int mpiNumX = std::sqrt(mpiSize);
    int mpiNumY = mpiNumX;
    int loops = 1;
    bool cpuonly = false;
    int workgroupsize = 64;
    bool giveOpenClDeviceName = false;
    int printMpiRank = -1;
    int haloWidth = 1;

    if(argc > 1) {
        for(int i = 1; i < argc; ++i) {

            if(argv[i] == std::string("-x") && i < argc-1)
                localDimX = atoi(argv[++i]);
            else if(argv[i] == std::string("-y") && i < argc-1)
                localDimY = atoi(argv[++i]);
            else if(argv[i] == std::string("-gpu") && i < argc-1)
                portionGPU = atof(argv[++i]);
            else if(argv[i] == std::string("-xy") && i < argc-1) {
                localDimX = atoi(argv[++i]);
                localDimY = localDimX;
            } else if(argv[i] == std::string("-mpix") && i < argc-1)
                mpiNumX = atoi(argv[++i]);
            else if(argv[i] == std::string("-mpiy") && i < argc-1)
                mpiNumY = atoi(argv[++i]);
            else if(argv[i] == std::string("-num") && i < argc-1)
                loops = atoi(argv[++i]);
            else if(argv[i] == std::string("-cpu"))
                cpuonly = true;
            else if(argv[i] == std::string("-wgs") && i < argc-1)
                workgroupsize = atof(argv[++i]);
            else if(argv[i] == std::string("-gpuinfo"))
                giveOpenClDeviceName = true;
            else if(argv[i] == std::string("-print") && i < argc-1)
                printMpiRank = atoi(argv[++i]);
            else if(argv[i] == std::string("-halo") && i < argc-1)
                haloWidth = atoi(argv[++i]);
        }
    }

    if(mpiRank == 0) {

        std::cout << std::endl
                  << "localDimX     = " << localDimX << std::endl
                  << "localDimY     = " << localDimY << std::endl
                  << "portionGPU    = " << portionGPU << std::endl
                  << "mpiNumX       = " << mpiNumX << std::endl
                  << "mpiNumY       = " << mpiNumY << std::endl
                  << "loops         = " << loops << std::endl
                  << "version       = " << (cpuonly ? "CPU-only" : "CPU/GPU") << std::endl
                  << "workgroupsize = " << workgroupsize << std::endl
                  << std::endl;

    }
    Sample sample(localDimX, localDimY, portionGPU, loops, haloWidth, mpiNumX, mpiNumY, cpuonly, workgroupsize, giveOpenClDeviceName);

    if(mpiRank == printMpiRank) {
        if(!cpuonly) {
            std::cout << "-------------------------------" << std::endl;
            std::cout << "-------------------------------" << std::endl;
            std::cout << "GPU region BEFORE" << std::endl;
            std::cout << "-------------------------------" << std::endl;
            sample.printGPU();
        }
        std::cout << "-------------------------------" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        std::cout << "CPU region BEFORE" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        sample.printCPU();
        std::cout << "-------------------------------" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::steady_clock::now();

    if(!cpuonly) {

        std::future<void> thrdGPU(std::async(std::launch::async, &Sample::launchGPU, &sample));
        sample.launchCPU();
        thrdGPU.wait();

    } else {

        sample.launchCPU();

    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();

    if(mpiRank == printMpiRank) {
        if(!cpuonly) {
            std::cout << "-------------------------------" << std::endl;
            std::cout << "-------------------------------" << std::endl;
            std::cout << "GPU region AFTER" << std::endl;
            std::cout << "-------------------------------" << std::endl;
            sample.printGPU();
        }
        std::cout << "-------------------------------" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        std::cout << "CPU region AFTER" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        sample.printCPU();
        std::cout << "-------------------------------" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(mpiRank == 0)
        std::cout << "Time required: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    MPI_Finalize();

    return 0;

}
