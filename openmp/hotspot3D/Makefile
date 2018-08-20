include ../../common/make.config

OUTPUT = output.txt

3D:
	$(CC) -o hotspot3D $(CFLAGS) hotspot3D.c -lm $(OPENMP_INC)

verify:
	$(CC) -o hotspot3D $(CFLAGS) hotspot3D.c -lm $(OPENMP_INC) -DVERIFY

clean:
	rm -f hotspot3D $(OUTPUT)
