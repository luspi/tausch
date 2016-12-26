#include "tausch.h"

Tausch::Tausch(int dim_x, int dim_y) {

    // get MPI info
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    mpi_dim_x = std::sqrt(mpi_size);
    mpi_dim_y = mpi_dim_x;

    // store configuration
    this->dim_x = dim_x;
    this->dim_y = dim_y;

    recvbuffer = nullptr;

    setupOpenCL();

}

Tausch::~Tausch() {
}

// Start creating sends and recvs for the halo data
void Tausch::startTausch() {

    // Get the current boundary data as one array
    double *cpuboundary = collectCPUBoundaryData();

    // Create an empty array for holding the received halo data
    if(recvbuffer != nullptr)
        delete[] recvbuffer;
    recvbuffer = new double[2*dim_x+2*dim_y + 4]{};

    // count how many send/recvs were created
    sendRecvCount = 0;
    sendRecvRequest = new MPI_Request[2*8];

    // send/recv to/from below
    if(mpi_rank > mpi_dim_x-1) {
        MPI_Isend(&cpuboundary[0], dim_x, MPI_DOUBLE, mpi_rank-mpi_dim_x, 0, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[0], dim_x, MPI_DOUBLE, mpi_rank-mpi_dim_x, 3, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

    // send/recv to/from left
    if(mpi_rank%mpi_dim_x != 0) {
        MPI_Isend(&cpuboundary[dim_x], dim_y, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[dim_x], dim_y, MPI_DOUBLE, mpi_rank-1, 2, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

    // send/recv to/from right
    if((mpi_rank+1)%mpi_dim_x != 0) {
        MPI_Isend(&cpuboundary[dim_x+dim_y], dim_y, MPI_DOUBLE, mpi_rank+1, 2, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[dim_x+dim_y], dim_y, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

    //send/recv to/from top
    if(mpi_rank < mpi_size-mpi_dim_x) {
        MPI_Isend(&cpuboundary[dim_x+2*dim_y], dim_x, MPI_DOUBLE, mpi_rank+mpi_dim_x, 3, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[dim_x+2*dim_y], dim_x, MPI_DOUBLE, mpi_rank+mpi_dim_x, 0, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

    // send/recv bottom left corner
    if(mpi_rank > mpi_dim_x-1 && mpi_rank%mpi_dim_x != 0) {
        MPI_Isend(&cpuboundary[0], 1, MPI_DOUBLE, mpi_rank-mpi_dim_x-1, 4, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[2*dim_x+2*dim_y], 1, MPI_DOUBLE, mpi_rank-mpi_dim_x-1, 6, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

    // send/recv bottom right corner
    if(mpi_rank > mpi_dim_x-1 && (mpi_rank+1)%mpi_dim_x != 0) {
        MPI_Isend(&cpuboundary[dim_x-1], 1, MPI_DOUBLE, mpi_rank-mpi_dim_x+1, 5, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[2*dim_x+2*dim_y+1], 1, MPI_DOUBLE, mpi_rank-mpi_dim_x+1, 7, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

    // send/recv top left corner
    if(mpi_rank < mpi_size-mpi_dim_x && mpi_rank%mpi_dim_x != 0) {
        std::cout << "bot left " << cpuboundary[dim_x+2*dim_y] << std::endl;
        MPI_Isend(&cpuboundary[dim_x+2*dim_y], 1, MPI_DOUBLE, mpi_rank+mpi_dim_x-1, 7, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[2*dim_x+2*dim_y+2], 1, MPI_DOUBLE, mpi_rank+mpi_dim_x-1, 5, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

    // send/recv top right corner
    if(mpi_rank < mpi_size-mpi_dim_x && (mpi_rank+1)%mpi_dim_x != 0) {
        MPI_Isend(&cpuboundary[2*dim_x+2*dim_y -1], 1, MPI_DOUBLE, mpi_rank+mpi_dim_x+1, 6, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount]);
        MPI_Irecv(&recvbuffer[2*dim_x+2*dim_y+3], 1, MPI_DOUBLE, mpi_rank+mpi_dim_x+1, 4, MPI_COMM_WORLD, &sendRecvRequest[2*sendRecvCount+1]);
        ++sendRecvCount;
    }

}

void Tausch::completeTausch() {

    // wait for all the local send/recvs to complete before moving on
    MPI_Waitall(2*sendRecvCount, sendRecvRequest, MPI_STATUSES_IGNORE);

    // distribute received data into halo regions
    distributeCPUHaloData();




    std::stringstream str;
    str << mpi_rank << " :: ";
    for(int i = 0; i < dim_x+2; ++i)
        str << cpudat[i] << "  ";
    str << std::endl;
    std::cout << str.str();


}

// function to collect the current boundary data and put it all into one array
double *Tausch::collectCPUBoundaryData() {

    double *ret = new double[2*dim_x+2*dim_y]{};

    // bottom
    if(mpi_rank < mpi_dim_x-1) {
        for(int i = 0; i < dim_x; ++i)
            ret[i] = cpudat[1+(dim_x+2) + i];
    }
    // left
    if(mpi_rank%mpi_dim_x != 0) {
        for(int i = 0; i < dim_y; ++i)
            ret[dim_x+i] = cpudat[1+(dim_x+2) + i*(dim_x+2)];
    }
    // right
    if((mpi_rank+1)%mpi_dim_x != 0) {
        for(int i = 0; i < dim_y; ++i)
            ret[dim_x+dim_y+i] = cpudat[2*(dim_x+2)-2 + i*(dim_x+2)];
    }
    // top
    if(mpi_rank < mpi_size-mpi_dim_x) {
        for(int i = 0; i < dim_x; ++i)
            ret[dim_x+2*dim_y+i] = cpudat[dim_y*(dim_x+2)+1 + i];
    }

    return ret;

}

// function to distribute the received boundary data into the right halo region
void Tausch::distributeCPUHaloData() {

    // bottom
    if(mpi_rank > mpi_dim_x-1) {
        for(int i = 0; i < dim_x; ++i)
            cpudat[1+i] = recvbuffer[i];
    }
    // left
    if(mpi_rank%mpi_dim_x != 0) {
        for(int i = 0; i < dim_y; ++i)
            cpudat[(i+1)*(dim_x+2)] = recvbuffer[dim_x+i];
    }
    // right
    if((mpi_rank+1)%mpi_dim_x != 0) {
        for(int i = 0; i < dim_y; ++i)
            cpudat[(i+2)*(dim_x+2)-1] = recvbuffer[dim_x+dim_y+i];
    }
    // top
    if(mpi_rank < mpi_size-mpi_dim_x) {
        for(int i = 0; i < dim_x; ++i)
            cpudat[(dim_y+1)*(dim_x+2)+1+i] = recvbuffer[dim_x+2*dim_y+i];
    }

    // bottom left corner
    if(mpi_rank > mpi_dim_x-1 && mpi_rank%mpi_dim_x != 0)
        cpudat[0] = recvbuffer[2*dim_x+2*dim_y];
    // bottom right corner
    if(mpi_rank > mpi_dim_x-1 && (mpi_rank+1)%mpi_dim_x != 0)
        cpudat[dim_x+2 -1] = recvbuffer[2*dim_x+2*dim_y + 1];
    // top left corner
    if(mpi_rank < mpi_size-mpi_dim_x && mpi_rank%mpi_dim_x != 0)
        cpudat[(dim_y+1)*(dim_x+2)] = recvbuffer[2*dim_x+2*dim_y + 2];
    // top right corner
    if(mpi_rank < mpi_size-mpi_dim_x && (mpi_rank+1)%mpi_dim_x != 0)
        cpudat[(dim_y+2)*(dim_x+2)-1] = recvbuffer[2*dim_x+2*dim_y + 3];

}

// Create OpenCL context and choose a device (if multiple devices are available, the MPI ranks will split up evenly)
void Tausch::setupOpenCL() {

    try {

        // Get platform count
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        int platform_length = all_platforms.size();

        // We need at most mpi_size many devices
        int *platform_num = new int[mpi_size]{};
        int *device_num = new int[mpi_size]{};

        // Counter so that we know when to stop
        int num = 0;

        // Loop over platforms
        for(int i = 0; i < platform_length; ++i) {
            // Get devices on platform
            std::vector<cl::Device> all_devices;
            all_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
            int device_length = all_devices.size();
            // Loop over platforms
            for(int j = 0; j < device_length; ++j) {
                // Store current pair
                platform_num[num] = i;
                device_num[num] = j;
                ++num;
                // and stop
                if(num == mpi_size) {
                    i = platform_length;
                    break;
                }
            }
        }

        // Get the platform and device to be used by this MPI thread
        cl_platform = all_platforms[platform_num[mpi_rank%num]];
        std::vector<cl::Device> all_devices;
        cl_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        cl_default_device = all_devices[device_num[mpi_rank%num]];

        // Give some feedback of the choice.
        std::cout << "Rank " << mpi_rank << " using OpenCL platform #" << platform_num[mpi_rank%num] << " with device #" << device_num[mpi_rank%num] << ": " << cl_default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

        delete[] platform_num;
        delete[] device_num;

        // Create context and queue
        cl_context = cl::Context({cl_default_device});
        cl_queue = cl::CommandQueue(cl_context,cl_default_device);

        // The OpenCL kernel
        std::ifstream cl_file("kernel.cl");
        std::string str;
        cl_file.seekg(0, std::ios::end);
        str.reserve(cl_file.tellg());
        cl_file.seekg(0, std::ios::beg);
        str.assign((std::istreambuf_iterator<char>(cl_file)), std::istreambuf_iterator<char>());

        // Create program and build
        cl_programs = cl::Program(cl_context, str, false);
        cl_programs.build("");

    } catch(cl::Error error) {
        std::cout << "[setup] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_default_device);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** " << std::endl << std::endl;
        }
        exit(1);
    }

}
