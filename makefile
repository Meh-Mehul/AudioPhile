EXEC = AudioBanger
CFLAGS = -Wall
SNDSTREAM = 2>/dev/null
CLIB = -lportaudio -lsndfile -lfftw3 -lm
DEP = src/Audio.cpp


$(EXEC): main.cpp
	@echo "making and Running the class implementation"
	@g++ $(CFLAGS) -o build/$@ $^ $(DEP) $(CLIB) 
	@build/$@ $(SNDSTREAM)

test: test.cpp
	@echo "Testing The Sound Hardware"
	@g++ $(CFLAGS) -o build/test $^ $(DEP) $(CLIB)
	@build/test $(SNDSTREAM)
.PHONY: test

install-deps:install-deps
	@echo "Installing Dependencies for AudioPhile..."
	@sudo apt install libsndfile1 libsndfile-dev libportaudio2 portaudio19-dev libportaudiocpp0 libfftw3-dev ffmpeg
	@echo "All Dependencies Installed."

run: 
	@echo "Running without warnings.."
	@./$(EXEC) 2>/dev/null
.PHONY:run

clean:
	rm -f build/$(EXEC)
	rm -f build/test
.PHONY: clean