typedef double real_t;

kernel void distributeHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const dimZ, global const int * restrict const haloWidth,
                               global real_t * vec, global const real_t * restrict const sync) {

    unsigned int current = get_global_id(0);
    int maxNum = 2*(*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth)) + 2*(*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth)) + 2*(*haloWidth)*(*dimX)*(*dimY);

    if(current >= maxNum)
        return;

    // left
    if(current < (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth))) {
        int index = ( current/(*haloWidth*(*dimY+2*(*haloWidth))) ) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimY+2*(*haloWidth)))) / (*haloWidth) ) * (*dimX+2*(*haloWidth)) +
                    current%(*haloWidth);
        vec[index] = sync[current];
        return;
    }
    int offset = (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth));
    // right
    if(current < offset + (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth))) {
        current -= offset;
        int index = ( current/(*haloWidth*(*dimY+2*(*haloWidth))) ) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimY+2*(*haloWidth)))) / (*haloWidth) ) * (*dimX+2*(*haloWidth)) +
                    current%(*haloWidth) + *dimX+*haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // top
    offset += (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth));
    if(current < offset + (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth))) {
        current -= offset;
        int index = ( current/(*haloWidth*(*dimX))) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimX))) / (*dimX)  + *dimY+*haloWidth) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX) + *haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // bottom
    offset += (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth));
    if(current < offset + (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth))) {
        current -= offset;
        int index = ( current/(*haloWidth*(*dimX))) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimX))) / (*dimX) ) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX) + *haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // front
    offset += (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth));
    if(current < offset + (*haloWidth)*(*dimX)*(*dimY)) {
        current -= offset;
        int index = ( current/((*dimX)*(*dimY))) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%((*dimX)*(*dimY))) / (*dimX) + *haloWidth) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX) + *haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // back
    offset += (*haloWidth)*(*dimX)*(*dimY);
    current -= offset;
    int index = ( current/((*dimX)*(*dimY)) + (*dimZ) + (*haloWidth)) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                ( (current%((*dimX)*(*dimY))) / (*dimX) + *haloWidth) * (*dimX+2*(*haloWidth)) +
                current%(*dimX) + *haloWidth;
    vec[index] = sync[offset+current];

}
