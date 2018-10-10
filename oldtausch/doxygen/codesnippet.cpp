int numBuffers = 2;
int msgtag = 0;

Tausch<double> *tausch = new Tausch2D<double>(MPI_DOUBLE, numBuffers, nullptr, MPI_COMM_WORLD);

TauschHaloSpec localHaloSpecs;

localHaloSpecs.haloX = 98;
localHaloSpecs.haloY = 0;
localHaloSpecs.haloWidth = 2;
localHaloSpecs.haloHeight = 100;
localHaloSpecs.bufferWidth = 100;
localHaloSpecs.bufferHeight = 100;
localHaloSpecs.remoteMpiRank = 0;

tausch->setLocalHaloInfo(TAUSCH_CwC, 1, localHaloSpecs);

tausch->postAllReceives(TAUSCH_CwC, &msgtag);

tausch->packSendBuffer(TAUSCH_CwC, 0, 0, buf1);
tausch->packSendBuffer(TAUSCH_CwC, 0, 1, buf2);
tausch->send(TAUSCH_CwC, 0, msgtag);

// receive the left buffers and unpack them
tausch->recv(TAUSCH_CwC, 0);
tausch->unpackRecvBuffer(TAUSCH_CwC, 0, 0, buf1);
tausch->unpackRecvBuffer(TAUSCH_CwC, 0, 1, buf2);
