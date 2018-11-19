CPP=g++
CPPFLAGS=-std=c++0x \
				 -Wall -g \

PROG=main
SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)

$(PROG): $(OBJS)
		$(CPP) -o $(PROG) $(OBJS)

clean:
		rm -f $(PROG) $(OBJS)
