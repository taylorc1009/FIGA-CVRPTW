compile:
	cl list.c data.c objective_function.c /link /out:objective_function.exe

clean:
	del *.exe
	del *.obj
	del *.o
	rmdir /s /q build

cmake:
	mkdir build
	cmake -B build