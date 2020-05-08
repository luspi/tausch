#ifndef TAUSCH_SETUP_OPENCL
#define TAUSCH_SETUP_OPENCL


static cl::Device tauschcl_device;
static cl::Context tauschcl_context;
static cl::CommandQueue tauschcl_queue;

static void setupOpenCL() {
    try {

        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        cl::Platform tauschcl_platform = all_platforms[0];

        std::vector<cl::Device> all_devices;
        tauschcl_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        tauschcl_device = all_devices[0];

        std::cout << "Using OpenCL device " << tauschcl_device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create context and queue
        tauschcl_context = cl::Context({tauschcl_device});
        tauschcl_queue = cl::CommandQueue(tauschcl_context,tauschcl_device);

    } catch(cl::Error &error) {
        std::cout << "[setup] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}


#endif
