all: build

build:
	@mkdir -p bin
	@gcc main.c -o bin/exe

run: build
	@./bin/exe
