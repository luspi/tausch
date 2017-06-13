kernel void packNextSendBuffer(global const int * restrict const gpuDim, global const size_t * restrict const haloSpecs,
                               global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const numBuffersPacked,
                               global double * restrict const haloBuffer, global const double * restrict const buffer) {

    const int current = get_global_id(0);

    int maxSize = haloSpecs[2]*haloSpecs[3];

    if(current >= maxSize) return;

    int index = (current/haloSpecs[2] + haloSpecs[1])*gpuDim[0] +
                 current%haloSpecs[2] + haloSpecs[0];

    for(int val = 0; val < valuesPerPointPerBuffer[*numBuffersPacked]; ++val) {
        int offset = 0;
        for(int b = 0; b < *numBuffersPacked; ++b)
            offset += valuesPerPointPerBuffer[b]*maxSize;
        haloBuffer[offset + valuesPerPointPerBuffer[*numBuffersPacked]*current + val] = buffer[valuesPerPointPerBuffer[*numBuffersPacked]*index + val];
    }

}

kernel void unpackNextRecvBuffer(global const int * restrict const gpuDim, global const size_t * restrict const haloSpecs,
                                 global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const numBuffersUnpacked,
                                 global const double * restrict const haloBuffer, global double * restrict const buffer) {

    const int current = get_global_id(0);

    int maxSize = haloSpecs[2]*haloSpecs[3];

    if(current >= maxSize) return;

    int index = (current/haloSpecs[2] + haloSpecs[1])*gpuDim[0] +
                 current%haloSpecs[2] + haloSpecs[0];

    for(int val = 0; val < valuesPerPointPerBuffer[*numBuffersUnpacked]; ++val) {
        int offset = 0;
        for(int b = 0; b < *numBuffersUnpacked; ++b)
            offset += valuesPerPointPerBuffer[b]*maxSize;
        buffer[valuesPerPointPerBuffer[*numBuffersUnpacked]*index + val] =
                haloBuffer[offset + valuesPerPointPerBuffer[*numBuffersUnpacked]*current + val];
    }

}

kernel void incrementBuffer(global int * restrict const buffer) {
    if(get_global_id(0) > 0) return;
    *buffer += 1;
}
