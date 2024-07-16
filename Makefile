C_FILES = utils/utils.c tensors/tensors.c tensors/matrix/matrix.c tensors/vector/vector.c linreg/linreg.c

all: build

build:
	@mkdir -p bin
	@gcc main.c $(C_FILES) -o bin/exe

run: build
	@./bin/exe

memcheck: build
	@ valgrind --leak-check=full ./bin/exe