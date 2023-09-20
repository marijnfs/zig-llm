all:
	zig build
release:
	zig build -Doptimize=ReleaseSafe
