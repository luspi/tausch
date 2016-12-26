CC = mpic++

# leave empty!
CFLAGS =

CFLAGS_ = -std=c++11 -Wall -Iinclude -lOpenCL
CFLAGS_ALL = -O3 -march=native
CFLAGS_DBG = -g
CFLAGS_OMP = -fopenmp

OUT = sample
OBJ_FILES = main.o tausch.o sample.o

all: CFLAGS = $(CFLAGS_) $(CFLAGS_ALL) $(CFLAGS_OMP)
all: $(OBJ_FILES)
	$(CC) -o $(OUT) $(CFLAGS) $^ $(LINK)

nomp: CFLAGS = $(CFLAGS_) $(CFLAGS_ALL)
nomp: $(OBJ_FILES)
	$(CC) -o $(OUT) $(CFLAGS) $^ $(LINK)

dbg: CFLAGS = $(CFLAGS_) $(CFLAGS_DBG)
dbg: $(OBJ_FILES)
	$(CC) -o $(OUT) $(CFLAGS) $^ $(LINK)

$(OBJ_FILES): %.o: %.cpp
	$(CC) -c $< $(CFLAGS) -o $@ $(LINK)

clean:
	$(RM) $(OBJ_FILES)

