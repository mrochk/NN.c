all: build

build:
	@mkdir -p bin
	@gcc main.c utils/utils.c tensors/matrix/matrix.c tensors/vector/vector.c -o bin/exe

run: build
	@./bin/exe

memcheck: build
	@ valgrind --leak-check=full ./bin/exe