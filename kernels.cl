kernel void collectHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                            global const double * restrict const vec, global double * sync) {

    unsigned int current = get_global_id(0);
    unsigned int maxNum = 2*(*dimX) + 2*(*dimY);

    if(current >= maxNum)
        return;

    // left
    if(current < *dimY) {
        sync[current] = vec[(1+current)*(*dimX+2) +1];
        return;
    }

    // right
    if(current < 2*(*dimY)) {
        sync[current] = vec[(2+(current-(*dimY)))*(*dimX+2) -2];
        return;
    }

    // top
    if(current < 2*(*dimY) + *dimX) {
        sync[current] = vec[(*dimX+2)*(*dimY)+1 + current-(2*(*dimY))];
        return;
    }

    // bottom
    sync[current] = vec[1+(*dimX+2)+(current-2*(*dimY)-(*dimX))];


}

kernel void distributeHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                               global double * vec, global const double * restrict const sync) {

    unsigned int current = get_global_id(0);
    unsigned int maxNum = 2*(*dimX+2) + 2*(*dimY);

    if(current >= maxNum)
        return;

    // left
    if(current < *dimY) {
        vec[(1+current)*(*dimX+2)] = sync[current];
        return;
    }

    // right
    if(current < 2*(*dimY)) {
        vec[(2+(current-(*dimY)))*(*dimX+2) -1] = sync[current];
        return;
    }

    // top
    if(current < 2*(*dimY)+(*dimX+2)) {
        vec[(*dimX+2)*(*dimY+1) + current-(2*(*dimY))] = sync[current];
        return;
    }

    // bottom
    vec[current-2*(*dimY)-(*dimX+2)] = sync[current];



}
