#include "src/Audio.h"
// Test File for AudioPhile Library
// Please run 'make test' first to set Device ID below
int deviceID = 7;


// Additional Note:
//      Since, sound buffers are large in size, due to compute constraints, memory management for audio files is hard 
//      and cannot be abstracted easily with a wrapper clas. I have tried to keep garbage collection as optimized as possible (it's still not the best)
//      but there still remains some issues. So, while working with AudioFiles, try to keep the operations on AudioFiles as close as possible (in code)
//      and try to instantiate new objects in heap.

int main() {
    try {
        
        // 1. Open a test sample (.wav format only)
        AudioFile audio("test.wav");
        // 2. Play the sample
        audio.playAudio(deviceID);
        // 3. Change Loudness Channel-Wise
        audio.changeLoudness(0.5, "L");
        // 4. Normalize the Audio Sample for Pre-Processing
        audio.normalizeAudio();
        // 5. Convert to Mono or Stereo
        audio.convertStereoToMono('L');
        // 6. Fasten and slower the audio sample
        audio.speedUp(1.3);
        // 7. Open and record a Blank Sample
        AudioFile blank(2, 44100, 512);
        blank.recordAudio(deviceID, 5);
        // 8. Save the file in .wav format
        blank.saveAudioFileasWAV("output.wav");
        // 9. Save the file as mp3 format
        blank.saveAudioFileasMP3("outputMp3");
        // 10. Reverse an AudioFile
        std::shared_ptr<AudioFile> reversed = audio.reverseAudio();
        // 11. Concatenate Audio Files
        AudioFile* concatted = ConcatAudio(&audio, &*reversed);
        // 12. Slicing Audio Files like Arrays
        std::shared_ptr<AudioFile> sliced = concatted->slice(1, 5);
        // 13. Get Batch FFT of the sound sample
        std::vector<float> batchFFT  = sliced->getbatchFFT(sliced->audioData);
        // 14. Moving Average, gaussian and Sharpen Kernels
        std::shared_ptr<AudioFile> filtered = sliced->applyFilter(MA_KERNEL);// others are:  GAUSSIAN_KERNEL, SHARPEN_KERNEL
        // 15. Functions to Generate Kernels
        std::vector<float> ker1 = CreateGaussianKernel(100, 0.2);
        std::vector<float> ker2 = CreateMovingAverageKernel(101);
        std::vector<float> ker3 = CreateSavitzkyGolayKernel(101); // These can be applied to the applyFilter method
        // 16. Function to De-Noise Audio (Still in-progress)
        std::shared_ptr<AudioFile> denoised = filtered->deNoise(200);
        // 17. Add any Audio as Sine Wave of base frequency and Amplitude
        blank.addAsSine(220, 5, 0.5);
        // 18. Modulate the audio with any sine wave
        blank.modulateWithSineWave(440, 0.8);
        // 19.Superimpose with any Sine Wave
        blank.superImposeSine(210, 0.7);
        // 20. Robot Voice ;)
        blank.recordAudio(deviceID, 5);
        blank.TurnMeIntoRobot();
        blank.playAudio(deviceID);

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

