all: build

build:
	@mkdir -p bin
	@gcc main.c linreg/linreg.c -o bin/exe

run: build
	@./bin/exe
