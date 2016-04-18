CC = gcc

CFLAGS 	=  -std=gnu99 -O3
HEADERS	= nnhelp.h
LIBS	= -lblas -lm

TARGETS	= nnhelp-test nn
OBJECTS = nnhelp.o nnhelp-test.o nn.o

all: 	$(TARGETS)

nnhelp-test: nnhelp-test.o nnhelp.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

nn: nn.o nnhelp.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) -o $@ $< $(LIBS)

check: clean nnhelp-test
	./nnhelp-test

clean :
	rm -rf $(TARGETS)
	rm -rf $(OBJECTS)
