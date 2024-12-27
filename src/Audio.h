#include <stdlib.h>
#include <stdio.h>
#include <portaudio.h>
#include <sndfile.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
// #define SAMPLE_RATE 44100
// #define FRAMES_PER_BUFFER 512
// #define NUM_CHANNELS 2
const std::vector<float> MA_KERNEL = {0.2, 0.2, 0.2, 0.2, 0.2};
const std::vector<float> GAUSSIAN_KERNEL = {1/16, 1/8, 1/4, 1/2, 1/4, 1/8, 1/16};
const std::vector<float> SHARPEN_KERNEL = {-0.5, -0.5, 1, -0.5, -0.5 };
class AudioFile
{
public:
    std::vector<float> audioData;
    char* filename;
    int NUM_CHANNELS= 2;
    int SAMPLE_RATE = 44100;
    int FRAMES_PER_BUFFER = 512;
    std::vector<const PaDeviceInfo*> devInfo;
    AudioFile(const char* filename); // Load from a File Constructor
    AudioFile(int channels, int sample_rate, int frames_per_buffer); // To be Recorded(empty) Constructor
    AudioFile(AudioFile& audio); // Copy some other already-in-code audio refrence
    ~AudioFile();

    /// @brief Getter for individual channel data, i prefer 2 channels for now;
    /// @param channel ```L``` or ```R```
    /// @return vector for that channel's data
    std::vector<float> getChannelAudiodata(char channel);
    /// @brief returns durations of audioSample in Seconds.
    float getDuration();

    // Function to get Available devices connected with system
    std::vector<const PaDeviceInfo*> getDevicesInfo();

    /// @brief This Function Plays the Sound in the (Non-empty) AudioFile , Currently only ```.wav``` is supported. 
    /// @param id Device ID of the playback Device, Check Compatibily using ```aplay -l``` or by printing the device from  ```this.getDeviceInfo()``` method
    void playAudio(PaDeviceIndex id);
    
    /// @brief This Function Records Sound in the ```audio``` varible in the   ```AudioFile``` class. Previously recorded sounds are overwritten
    /// @param id Device ID of the playback Device, Check Compatibily using ```aplay -l``` or by printing the device from  ```this.getDeviceInfo()``` method
    /// @param timeinSeconds Time for which recording should be done, Note that the recording is Synchronous in Nature
    void recordAudio(PaDeviceIndex id, int timeinSeconds);

    /// @brief This function Saves the Audio file as a WAV File in the same directoty with the given or pre-defined name
    /// @param filename The name of the file ```With the .wav extension , example "hello.wav"```
    void saveAudioFileasWAV(const char* filename); 

    /// @brief Saves the soundFile in ```.mp3``` format (for dumb reasons it also creates a .wav file)
    /// @param filename Name of the file or null if already defined ```DO NOT USE .mp3 with the file```
    void saveAudioFileasMP3(std::string filename);

    /// @brief This function reverses the contents of the AudioFile
    /// @return returns a shared pointer to the new reversed AudioFile object
    std::shared_ptr<AudioFile> reverseAudio();

    /// @brief This Function Changes the Loudness of the audioFile by a certain gain. Note that the loudness may not be increased after a certain factor due to PCM issues
    /// @param gain the factor that the sound should be increased by (>=0)
    /// @param channel the channel you want the loudness to be affected of. ```L```, ```R```, and deafult is ```LR```.
    void changeLoudness(float gain, std::string channel );

    /// @brief 🤔 maybe someone will use it for Model Quantization?(not me)
    void normalizeAudio();

    /// @brief Function to convert 1 channel audio, to dual channel audio
    void convertMonoToStereo();

    /// @brief function to convert Setreo sound to Mono. This is not same as ```isolateChannel``` method.
    /// @param channel The channel to keep as final, or merger. "L", "R", "M"
    void convertStereoToMono(char channel);

    /// @brief function to speed up the audio file. Current implementation based on resampling.
    /// @param factor factor of speedup should be ```>=1```
    void speedUp(float factor);

    /// @brief function to slow down the audio file. Current implementation based on interpolation resampling.
    /// @param factor factor of speedup should be ```>=1``` 
    void slowDown(float factor);


    /// @brief function to slice the sound files like arrays!
    /// @param startSecond starting second for the new audio sample, must be >=0 and can be a decimal upto 2nd position accuracy.
    /// @param stopSecond ending second for audio sample, if out of bounds, its automatically adjuted to be ending of the audio sample.
    /// @return shared_ptr to the new Audio sample, the shared_ptr approach allows parallel multi-batch processing due to automatic garabage collection. 
    std::shared_ptr<AudioFile> slice(float startSecond, float stopSecond);

    /// @brief function to get Fast Fourier Transform a batch of audio samples of length atmost 5 Seconds. Limit may be set by level of compute present. May be changed in the FFT_BATCH_LIMIT.
    /// @return This function returns the output of a 1-dimensional Real-to-Half-Complex Fast Fourier Transform.
    std::vector<float> getbatchFFT(std::vector<float> &batch);

    /// @brief Function to apply any kernel to the audio Signal. 
    /// @param kernel The kernel to be applied.
    /// @param optimize set it to true for faster runtime.
    /// @return New Audio File with Filtered Audio.
    std::shared_ptr<AudioFile> applyFilter(std::vector<float> kernel, bool optimize = true);

    /// @brief Its supposed to Denoise the sound. Doesnt work as intended as of now.
    /// @param threshold The Frequency thershold above which we need to keep stuff.
    /// @return New Audio Files with Supposed Denoised audio.
    std::shared_ptr<AudioFile> deNoise(float threshold);

    /// @brief This Function Adds the current sound as a base sine wave.
    /// @param frequency Frequency of desired signal in hertz.
    /// @param duration Durection for which it needs to be added in seconds.
    /// @param amplitude Amplitude of sine wave, must be between -1 and 1.
    void addAsSine(float frequency, float duration,float amplitude = 0.5);

    /// @brief Function to Superimpose current audio with a sine Wave
    /// @param frequency Frequency of desired signal in hertz.
    /// @param amplitude Amplitude of sine wave, must be between -1 and 1.
    void superImposeSine(float frequency,float amplitude);

    /// @brief Function to Modulate current audio with a sine Wave
    /// @param frequency Frequency of desired signal in hertz.
    /// @param amplitude Amplitude of sine wave, must be between -1 and 1.
    void modulateWithSineWave(float frequency, float amplitude);

    /// @brief Turns the sound into robotic Sound.
    void TurnMeIntoRobot();
};

/// @brief Merges the audio content in one file after the another
/// @param audio1 first file to appear at start
/// @param audio2 second file whose content is to be appended
/// @return pointer to the new merged AudioFile instance
AudioFile* ConcatAudio(AudioFile* audio1, AudioFile* audio2);


/// @brief Function to Create a Kernel for Audio Processing. Refrence: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
/// @param windowSize Size of Kernel
/// @return Kernel as a std::vector<float>
std::vector<float> CreateSavitzkyGolayKernel(int windowSize);

/// @brief Function to Create a Kernel for Audio Processing
/// @param windowSize Size of Kernel
/// @return Kernel as a std::vector<float>
std::vector<float> CreateMovingAverageKernel(int windowSize);

/// @brief Function to Create a Kernel for Audio Processing
/// @param windowSize Size of Kernel
/// @param sigma Standard deviation of the normal Dist, (Flatness of the kernel)
/// @return Kernel as a std::vector<float>
std::vector<float> CreateGaussianKernel(int windowSize, float sigma);
