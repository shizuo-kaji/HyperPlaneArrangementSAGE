CXX := c++
CXXFLAGS := -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra -Wpedantic -pthread
TARGET := solver
SRC := solver.cpp
INPUT ?= example_input.json
OUTPUT ?=

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
ifneq ($(strip $(OUTPUT)),)
	./$(TARGET) $(INPUT) $(OUTPUT)
else
	./$(TARGET) $(INPUT)
endif

clean:
	rm -f $(TARGET)
