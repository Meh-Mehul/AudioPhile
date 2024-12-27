// Sound Library for Audio Processing in C++
// Meant for:
//          1. Deep Learning and ML preprocessing tasks
//          2. Audio recording and editing through code
//          3. Fun

#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <portaudio.h>
#include <sndfile.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <memory>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include "Audio.h"

// Some Constants we can alter
float FFT_BATCH_LIMIT = 5.0f;


// Some Constant i needed
int numChannels = 0;

// A very important Function for checking error in calls to Sound API
static void CheckErr(PaError err){
    if(err != paNoError){
        printf("Port Aduio Error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

// Base AudioPhile Class, This contains almost all functions
AudioFile::AudioFile(const char* filename){
    if (!filename || std::strlen(filename) == 0) {
    throw std::invalid_argument("Invalid filename\n");
    }
    SF_INFO sfInfo = {0}; 
    SNDFILE* sndFile = sf_open(filename, SFM_READ, &sfInfo);
    if (!sndFile) {
        throw std::runtime_error(sf_strerror(nullptr));
    }
    this->NUM_CHANNELS = sfInfo.channels;
    numChannels = this->NUM_CHANNELS;
    this->SAMPLE_RATE = sfInfo.samplerate;
    // this->FRAMES_PER_BUFFER = sfInfo.frames;
    size_t numFrames = sfInfo.frames;
    audioData.resize(numFrames * NUM_CHANNELS);
    sf_count_t readCount = sf_read_float(sndFile, audioData.data(), audioData.size());
    if (readCount != static_cast<sf_count_t>(audioData.size())) {
        sf_close(sndFile);
        throw std::runtime_error("Failed to read all audio samples.\n");
    }
    sf_close(sndFile);    
    // this->getDevicesInfo();

}

AudioFile::AudioFile(int channels, int sample_rate, int frames_per_buffer){
    this->NUM_CHANNELS = channels;
    numChannels = channels;
    this->SAMPLE_RATE = sample_rate;
    this->FRAMES_PER_BUFFER = frames_per_buffer;
    this->getDevicesInfo();
}
AudioFile::AudioFile(AudioFile& audio){
    this->audioData = audio.audioData;
    this->NUM_CHANNELS = audio.NUM_CHANNELS;
    this->FRAMES_PER_BUFFER = audio.FRAMES_PER_BUFFER;
    this->filename = audio.filename;
    this->SAMPLE_RATE =audio.SAMPLE_RATE;
    this->getDevicesInfo();
}

AudioFile::~AudioFile(){
    audioData.clear();
    audioData.resize(0);
    // delete this;
}


std::vector<const PaDeviceInfo*> AudioFile::getDevicesInfo(){
    PaError err;
    err = Pa_Initialize();
    CheckErr(err);
    int numDevices = Pa_GetDeviceCount();
    if(numDevices < 0 ){
        printf("Error Getting Device Count\n");
        exit(EXIT_FAILURE);
    }
    else if(numDevices == 0){
        printf("No Devices Connected\n");
        exit(EXIT_SUCCESS);
    }
    std::vector<const PaDeviceInfo*> devices;
    for(int i = 0; i<numDevices; ++i){
        const PaDeviceInfo* dev = Pa_GetDeviceInfo(i);
        devices.push_back(dev);
    }
    this->devInfo = devices;
    return devices;
}

void AudioFile::playAudio(PaDeviceIndex id){
    // if(this->devInfo.size() == 0){
    if(this->audioData.size() == 0){
        printf("Please Call Play method on a non-empty AudioFile, You might wanna record first\n");
        exit(EXIT_FAILURE);
    }
    this->getDevicesInfo();
    if(id>this->devInfo.size()-1){
        printf("Device Index out of Range, Please Check the Device Connection First\n");
        exit(EXIT_FAILURE);
    }
    PaStream* stream;
    PaStreamParameters outputParams;
    memset(&outputParams, 0, sizeof(outputParams));
    outputParams.device = id;
    if (outputParams.device == paNoDevice) {
        std::cerr << "Not a valid output device. Some Error Occured ;(" << std::endl;
        Pa_Terminate();
        return;
    }
    outputParams.channelCount = this->NUM_CHANNELS;
    outputParams.sampleFormat = paFloat32; 
    outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;
    // printf("Playing Audio ...");
    PaError err  = Pa_OpenStream(
        &stream,
        nullptr,
        &outputParams,
        this->SAMPLE_RATE,
        this->FRAMES_PER_BUFFER,
        paClipOff,
        [](const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
           const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) -> int {
            auto* audioFile = static_cast<AudioFile*>(userData);
            static size_t playbackPosition = 0;
            float* out = static_cast<float*>(outputBuffer);
            size_t remainingSamples = audioFile->audioData.size() - playbackPosition;
            size_t samplesToWrite = framesPerBuffer * audioFile->NUM_CHANNELS;
            if (remainingSamples < samplesToWrite) {
                samplesToWrite = remainingSamples;
            }
            std::memcpy(out, &audioFile->audioData[playbackPosition], samplesToWrite * sizeof(float)); // We write to o/p buffer to play sound, Device API captures this stream automatically
            playbackPosition += samplesToWrite;
            // If we've played all audio data, stop the stream
            if (playbackPosition >= audioFile->audioData.size()) {
                playbackPosition = 0;
                return paComplete; 
            }
            return paContinue;
        },
        this
    );
    CheckErr(err);
    err = Pa_StartStream(stream);
    CheckErr(err);
    while (Pa_IsStreamActive(stream) == 1) {
        Pa_Sleep(100);
    }
    err = Pa_StopStream(stream);
    CheckErr(err);
    err = Pa_CloseStream(stream);
    CheckErr(err);
    Pa_Terminate();
    printf("Playback Complete\n");
    return;
}
int recordCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    std::vector<float>* output = (std::vector<float>*)userData;
    const float* input = (const float*)inputBuffer;
    for (unsigned long i = 0; i < framesPerBuffer * numChannels; ++i) {
        output->push_back(input[i]);
    }
    return paContinue;
}
void AudioFile::recordAudio(PaDeviceIndex id, int timeInSeconds){
    this->getDevicesInfo();
    if(id>this->devInfo.size()-1){
        printf("Device Index out of Range, Please Check the Device Connection First\n");
        exit(EXIT_FAILURE);
    }
    PaStreamParameters inputParameters;
    memset(&inputParameters, 0, sizeof(inputParameters));
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.device = id;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(id)->defaultLowInputLatency;

    PaStream* stream;
    std::vector<float> Data;   
    PaError err = Pa_OpenStream(
        &stream,
        &inputParameters,
        nullptr, 
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        recordCallback,
        &Data
    );
    CheckErr(err);
    

    err = Pa_StartStream(stream);
    CheckErr(err);

    printf("Recording for %d Seconds\n", timeInSeconds);
    Pa_Sleep(timeInSeconds*1000);

    err = Pa_StopStream(stream);
    CheckErr(err);

    err = Pa_CloseStream(stream);
    CheckErr(err);
    this->audioData = Data;
    printf("recorded the Sound, Please call .Save for Saving the sound\n");
}

void AudioFile::saveAudioFileasWAV(const char* filename){
    if (!this->filename || std::strlen(filename) == 0) {
    throw std::invalid_argument("Invalid filename\n");
    }
    if(this->audioData.size() == 0){
        printf("Please Call Play method on a non-empty AudioFile, You might wanna record first\n");
        exit(EXIT_FAILURE);
    }
    SF_INFO sfInfo;
    memset(&sfInfo, 0, sizeof(SF_INFO));
    sfInfo.samplerate = SAMPLE_RATE;
    sfInfo.channels = NUM_CHANNELS;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* outFile = sf_open(filename, SFM_WRITE, &sfInfo);
    if (!outFile) {
        printf("Error opening output file: %s\n", sf_strerror(outFile));
        exit(EXIT_FAILURE);
    }
    std::vector<short> audioData16Bit;
    for (size_t i = 0; i < audioData.size(); ++i) {
        audioData16Bit.push_back(static_cast<short>(this->audioData[i] * 32767.0f));  // Convert float to 16-bit
    }
    sf_count_t numFramesWritten = sf_write_short(outFile, audioData16Bit.data(), audioData16Bit.size());
    if (numFramesWritten != audioData16Bit.size()) {
        printf("Error writing audio data to file\n");
        sf_close(outFile);
        exit(EXIT_FAILURE);
    }

    sf_close(outFile);
    printf("Audio File Written to: %s\n", filename);
}


void AudioFile::saveAudioFileasMP3(std::string filename){
    const char* fwav = (filename + ".wav").c_str();
    this->saveAudioFileasWAV(fwav);
    std::string mp3File = filename + ".mp3";
    std::string command = "echo y | ffmpeg -i " + std::string(fwav) + " " + mp3File;
    int result = std::system(command.c_str());
    if (result == 0) {
        std::cout << "Conversion successful: " << fwav << " to " << mp3File << std::endl;
    } else {
        std::cerr << "Error during conversion." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string cmd = "rm -f " + std::string(fwav);
    result = std::system(cmd.c_str());
    if (result == 0) {
        return;
    } else {
        std::cerr << "Error during conversion." << std::endl;
        exit(EXIT_FAILURE);
    }    

}

std::shared_ptr<AudioFile> AudioFile::reverseAudio(){
    // AudioFile* newAudio = new AudioFile(*this);
    std::shared_ptr<AudioFile> newAudio = std::make_shared<AudioFile>(*this);
    reverse(newAudio->audioData.begin(), newAudio->audioData.end());
    return newAudio;
}



std::vector<float> AudioFile::getChannelAudiodata(char channel){
    if(this->NUM_CHANNELS != 2){
        printf("We cant process audio file with more than 2 channels for now, might do it later ;(\n");
        exit(EXIT_FAILURE);
    }
    std::vector<float> output;
    if(channel == 'L'){
        for(unsigned long i = 0; i<this->audioData.size(); i+=2){
            output.push_back(this->audioData[i]);
        }
    }
    else if(channel == 'R'){
        for(unsigned long i = 1; i<this->audioData.size(); i+=2){
            output.push_back(this->audioData[i]);
        }
    }
    else{
        printf("Channel should be L or R\n");
        exit(EXIT_FAILURE);
    }
    return output;
}

float AudioFile::getDuration(){
    return (float)audioData.size()/(float)this->SAMPLE_RATE;
}

void AudioFile::changeLoudness(float gain,std::string channel){
    if(gain <0.0){
        printf("Gain should be positive, although i dont know what happens if its negative, for now we dont change the loudness.\n");
        exit(EXIT_SUCCESS);
    }
    if(channel == "L"){
        if(this->NUM_CHANNELS != 2){
            printf("The Chanels in the audio should be 2 for selective loudness chnaging\n");
            exit(EXIT_FAILURE);
        }
        for(unsigned long i  = 0; i<this->audioData.size(); i += 2){
            // this->audioData[i] = std::min((float)1, this->audioData[i]*gain);
            if(this->audioData[i]>0.0){
                this->audioData[i] = std::min((float)1, this->audioData[i]*gain);
            }
            else{
                this->audioData[i] = std::max((float)-1, this->audioData[i]*gain);
            }
        }
    }
    else if(channel == "R"){
        if(this->NUM_CHANNELS != 2){
            printf("The Chanels in the audio should be 2 for selective loudness changing\n");
            exit(EXIT_FAILURE);
        }
        for(unsigned long i  = 1; i<this->audioData.size(); i += 2){
            // this->audioData[i] = std::min((float)1, this->audioData[i]*gain);
            if(this->audioData[i]>0.0){
                this->audioData[i] = std::min((float)1, this->audioData[i]*gain);
            }
            else{
                this->audioData[i] = std::max((float)-1, this->audioData[i]*gain);
            }
        }
    }
    else{
        for(unsigned long i  = 0; i<this->audioData.size(); ++i){
            // this->audioData[i] = std::min((float)1, this->audioData[i]*gain);
            if(this->audioData[i]>0.0){
                this->audioData[i] = std::min((float)1, this->audioData[i]*gain);
            }
            else{
                this->audioData[i] = std::max((float)-1, this->audioData[i]*gain);
            }
        }
    }
}

float getMean(std::vector<float> nums){
    float res = 0;
    for(auto i:nums) res += i;
    res /= (float)nums.size();
    return res;
}

float getStd(std::vector<float> nums){
    float mean = getMean(nums);
    float s2 = 0;
    for(auto i: nums){
        float p2 = pow(i-mean, 2);
        s2 += p2;
    }
    s2 /= nums.size();
    s2 = sqrtf(s2);
    return s2;
}

void AudioFile::normalizeAudio(){ 
    if(this->NUM_CHANNELS != 2){
        printf("Please convert to Stereo for Normalization.\n");
        exit(EXIT_FAILURE);
    }
    float ml, sl;
    std::vector<float> lchannel = getChannelAudiodata('L');
    ml = getMean(lchannel);
    sl = getStd(lchannel);
    for(unsigned long i = 0; i<this->audioData.size(); i+=2){
        float curr = this->audioData[i]; curr = (curr-ml)/sl;
        if(curr>1.0) curr = 1.0;
        if(curr <-1.0) curr = -1.0; 
        this->audioData[i] = curr;
    }
    lchannel.clear();
    float mr, sr;
    std::vector<float> rchannel = getChannelAudiodata('R');
    mr = getMean(rchannel); sr = getStd(rchannel);
    //printf("%f, %f, %f, %f", ml, mr, sl, sr);
    for(unsigned long i = 1; i<this->audioData.size(); i+=2){
        float curr = this->audioData[i]; curr = (curr-mr)/sr;
        if(curr>1.0) curr = 1.0;
        if(curr <-1.0) curr = -1.0;
        this->audioData[i] = curr;
    } 
    rchannel.clear(); 
}


void AudioFile::convertMonoToStereo(){
    if(this->NUM_CHANNELS != 1){
        printf("Can only convert mono to stereo using this function.\n");
        exit(EXIT_FAILURE);
    }
    this->NUM_CHANNELS = 2;
    std::vector<float> newvec;
    for(unsigned long i = 0; i<this->audioData.size();++i){
        float curr = this->audioData[i];
        newvec.push_back(curr);
        newvec.push_back(curr);
    }
    this->audioData = newvec;
}


void AudioFile::convertStereoToMono(char channel){
    if(this->NUM_CHANNELS != 2){
        printf("NUM_CHANNELS should be 2 for converting over to Mono\n");
        exit(EXIT_FAILURE);
    }
    std::vector<float> newvec;
    this->NUM_CHANNELS = 1;
    if(channel == 'L'){
        for(unsigned long i = 0; i<this->audioData.size(); i += 2){
            newvec.push_back(this->audioData[i]);
        }
    }
    else if(channel == 'R'){
        for(unsigned long i =1; i<this->audioData.size(); i += 2){
            newvec.push_back(this->audioData[i]);
        }
    }
    else if(channel == 'M'){
        for(unsigned long i =0; i<this->audioData.size(); i += 2){
            newvec.push_back((this->audioData[i]+this->audioData[i+1])/2);
        }
    }
    else{
        printf("Channel should be L or R or M.\n");
        exit(EXIT_FAILURE);
    }
    this->audioData = newvec;
}


void AudioFile::speedUp(float factor) {
    if (factor <= 1.0f) {
        throw std::invalid_argument("Speed-up factor must be greater than 1.");
    }
    std::vector<float> newAudioData;
    newAudioData.reserve(static_cast<size_t>(audioData.size() / factor));
    // Resample by skipping samples
    for (size_t i = 0; i < audioData.size(); i += static_cast<size_t>(factor)) {
        newAudioData.push_back(audioData[i]);
    }
    audioData = std::move(newAudioData);
    this->SAMPLE_RATE = static_cast<int>(this->SAMPLE_RATE * factor); // Adjust the sampling rate
}


void AudioFile::slowDown(float factor) {
    if (factor < 1.0f) {
        throw std::invalid_argument("Slow-down factor must be greater than 1.");
    }
    std::vector<float> newAudioData;
    newAudioData.reserve(static_cast<size_t>(audioData.size() * factor));

    // Resample by interpolating samples
    for (size_t i = 0; i < audioData.size() - 1; ++i) {
        newAudioData.push_back(audioData[i]);
        for (float f = 1.0f; f < factor; ++f) {
            // Linear interpolation
            float interpolated = audioData[i] + f * (audioData[i + 1] - audioData[i]) / factor;
            newAudioData.push_back(interpolated);
        }
    }
    newAudioData.push_back(audioData.back());
    audioData = std::move(newAudioData);
    this->SAMPLE_RATE = static_cast<int>(this->SAMPLE_RATE / factor); 
    while(audioData.size()%this->SAMPLE_RATE != 0){
        audioData.pop_back();
    }
}


std::shared_ptr<AudioFile> AudioFile::slice(float startSecond, float stopSecond){
    if(startSecond<0.0f){
        std::invalid_argument("Start Second should be >=0");
    }
    float time_dur = (float)this->audioData.size()/(float)this->SAMPLE_RATE;
    stopSecond = std::min(stopSecond, time_dur);
    unsigned long startSample = floor(startSecond*this->SAMPLE_RATE);
    unsigned long stopSample = floor(stopSecond*this->SAMPLE_RATE);
    std::shared_ptr<AudioFile> newfile = std::make_unique<AudioFile>(this->NUM_CHANNELS, this->SAMPLE_RATE, this->FRAMES_PER_BUFFER);
    std::vector<float>* ptr = &newfile->audioData;
    for(unsigned long i = startSample; i<stopSample; ++i){
        ptr->push_back(this->audioData[i]);
    }
    return newfile;
}



/*std::vector<std::pair<double, double>> AudioFile::getFFT(){*/
/*    size_t N = this->audioData.size();*/
/*    double* in = fftw_alloc_real(N);*/
/*    fftw_complex* out = fftw_alloc_complex(N / 2 + 1);*/
/*    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);*/
/*    for (size_t i = 0; i < N; ++i) {*/
/*        in[i] = static_cast<double>(this->audioData[i]);*/
/*    }*/
/*    fftw_execute(plan);*/
/*    printf("Dawg\n");*/
/*    std::vector<std::pair<double, double>> outputFFT;*/
/*    for (size_t i = 0; i < N / 2 + 1; ++i) {*/
/*        outputFFT.push_back(std::make_pair(out[i][0], out[i][1])); */
/*    }    */
/*    fftw_destroy_plan(plan);*/
/*    fftw_free(in);*/
/*    fftw_free(out);*/
//   return outputFFT; }

std::vector<float> AudioFile::getbatchFFT(std::vector<float> &batch){
    if((float)batch.size()/(float)this->SAMPLE_RATE > FFT_BATCH_LIMIT){
        printf("The batch samples must at most 5 seconds long. This limit may be set in the source file in the FFT_BATCH_LIMIT variable.\n");
        exit(EXIT_FAILURE);
    }
    size_t N = batch.size();

    double* in = fftw_alloc_real(N);
    double* out = fftw_alloc_real(N);
    for(size_t i = 0; i<N; i++){
        in[i] = static_cast<double>(batch[i]);
    }
    fftw_plan plan = fftw_plan_r2r_1d(N, in, out , FFTW_R2HC, FFTW_ESTIMATE);
    fftw_execute(plan);
    std::vector<float> output(out, out+N);
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return output;
}


std::vector<float> conv1d_brute_force(const std::vector<float>& signal, const std::vector<float>& kernel) {
    std::vector<float> result(signal.size() + kernel.size() - 1, 0.0f);
    for (size_t i = 0; i < signal.size(); i++) {
        for (size_t j = 0; j < kernel.size(); j++) {
            result[i + j] += signal[i] * kernel[j];
        }
    }
    return result;
}

bool isPowerOf2(size_t n) {
    return (n != 0) && ((n & (n - 1)) == 0);
}
size_t nextPowerOf2(size_t n) {
    size_t power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}
size_t bitReverse(size_t x, size_t log2n) {
    size_t n = 0;
    for (size_t i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}
std::vector<std::complex<float>> fft(std::vector<std::complex<float>> x) {
    size_t n = x.size();
    if (!isPowerOf2(n)) {
        size_t new_size = nextPowerOf2(n);
        x.resize(new_size, std::complex<float>(0, 0));
        n = new_size;
    }
    size_t log2n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) {
        log2n++;
    }
    for (size_t i = 0; i < n; i++) {
        size_t j = bitReverse(i, log2n);
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
    // CT Butterfly
    for (size_t len = 2; len <= n; len <<= 1) {
        float angle = -2 * M_PI / len;
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1, 0);
            for (size_t j = 0; j < len/2; j++) {
                std::complex<float> u = x[i + j];
                std::complex<float> t = w * x[i + j + len/2];
                x[i + j] = u + t;
                x[i + j + len/2] = u - t;
                w *= wlen;
            }
        }
    }
    return x;
}
std::vector<std::complex<float>> ifft(std::vector<std::complex<float>> x) {
    // Conjugate input
    for (auto& val : x) {
        val = std::conj(val);
    }
    x = fft(x);
    float scale = 1.0f / x.size();
    for (auto& val : x) {
        val = std::conj(val) * scale;
    }
    return x;
}
std::vector<float> conv1d_fft(const std::vector<float>& signal, const std::vector<float>& kernel) {
    size_t n = signal.size() + kernel.size() - 1;
    size_t N = nextPowerOf2(n);
    std::vector<std::complex<float>> signal_complex(N);
    std::vector<std::complex<float>> kernel_complex(N);
    for (size_t i = 0; i < signal.size(); i++) {
        signal_complex[i] = std::complex<float>(signal[i], 0);
    }
    for (size_t i = 0; i < kernel.size(); i++) {
        kernel_complex[i] = std::complex<float>(kernel[i], 0);
    }
    auto fft_signal = fft(signal_complex);
    auto fft_kernel = fft(kernel_complex);
    std::vector<std::complex<float>> product(N);
    for (size_t i = 0; i < N; i++) {
        product[i] = fft_signal[i] * fft_kernel[i];
    }
    auto result_complex = ifft(product);
    std::vector<float> result(n);
    for (size_t i = 0; i < n; i++) {
        result[i] = result_complex[i].real();
    }
    return result;
}


std::shared_ptr<AudioFile> AudioFile::applyFilter(std::vector<float> kernel, bool optimize){
    if(this->NUM_CHANNELS == 2){
        std::shared_ptr<AudioFile> newAduio = std::make_shared<AudioFile>(this->NUM_CHANNELS, this->SAMPLE_RATE, this->FRAMES_PER_BUFFER);
        auto ptr = &newAduio->audioData;
        std::unique_ptr<std::vector<float>> lchannel = std::make_unique<std::vector<float>>(this->getChannelAudiodata('L'));
        std::unique_ptr<std::vector<float>> rchannel = std::make_unique<std::vector<float>>(this->getChannelAudiodata('R'));
        if(optimize){
            std::vector<float> lfft = conv1d_fft(*lchannel, kernel);
            std::vector<float> rfft = conv1d_fft(*rchannel, kernel);
            for(int i = 0; i<lfft.size();++i){
                ptr->push_back(lfft[i]);
                ptr->push_back(rfft[i]);
            }
        }
        else{
            std::vector<float> lfft = conv1d_brute_force(*lchannel, kernel);
            std::vector<float> rfft = conv1d_brute_force(*rchannel, kernel);
            for(int i = 0; i<lfft.size();++i){
                ptr->push_back(lfft[i]);
                ptr->push_back(rfft[i]);
            }
        }
        return newAduio;
    }
    else if(this->NUM_CHANNELS == 1){
        std::shared_ptr<AudioFile> newAduio = std::make_shared<AudioFile>(this->NUM_CHANNELS, this->SAMPLE_RATE, this->FRAMES_PER_BUFFER);
        auto ptr = &newAduio->audioData;
        if(optimize){
            std::vector<float> lfft = conv1d_fft(this->audioData, kernel);
            for(int i = 0; i<lfft.size();++i){
                ptr->push_back(lfft[i]);            
            }
        }
        else{
            std::vector<float> lfft = conv1d_brute_force(this->audioData, kernel);
            for(int i = 0; i<lfft.size();++i){
                ptr->push_back(lfft[i]);
            }
        }
        return newAduio;
    }
    else{
        printf("Currently upto 2 channel Audio is Supported for Filter.\n");
        exit(EXIT_FAILURE);
    }
}


std::shared_ptr<AudioFile> AudioFile::deNoise(float threshold){
    float index_threshold = (float)(audioData.size()*threshold)/(float)SAMPLE_RATE;
    size_t n = this->audioData.size() - 1;
    size_t N = nextPowerOf2(n);
    std::vector<std::complex<float>> signal_complex(N);
    for (size_t i = 0; i < this->audioData.size(); i++) {
        signal_complex[i] = std::complex<float>(this->audioData[i], 0);
    }
    auto fft_signal = fft(signal_complex);
    for(unsigned long i = 0; i<std::min((int)fft_signal.size(), (int)index_threshold);++i){
        fft_signal[i] = 0;
    }
    auto denoised = ifft(fft_signal);
    std::shared_ptr<AudioFile> deNoised = std::make_unique<AudioFile>(this->NUM_CHANNELS, this->SAMPLE_RATE, this->FRAMES_PER_BUFFER);
    for(int i = 0; i<denoised.size();++i){
        deNoised->audioData.push_back((float)denoised[i].real());
    }
    return deNoised;
}


void AudioFile::addAsSine(float frequency, float duration,float amplitude) { 
    if(amplitude > 1.0f || amplitude < -1.0f){
        throw std::invalid_argument("The Amplitude of sine must be between -1 and 1.\n");
    }
    int totalSamples = static_cast<int>(duration * SAMPLE_RATE);
    audioData.resize(totalSamples); 
    for (int i = 0; i < totalSamples; ++i) {
        float sampleValue = amplitude * sinf(2.0f * M_PI * frequency * i / SAMPLE_RATE);
        audioData[i] = sampleValue; 
    } 
}

void AudioFile::superImposeSine(float frequency,float amplitude){
    int totalSamples = audioData.size();
    for (int i = 0; i < totalSamples; ++i) { 
        float carrier = amplitude * sinf(2.0f * M_PI * frequency * i / SAMPLE_RATE);
        audioData[i] += carrier; 
    } 
}
void AudioFile::modulateWithSineWave(float frequency, float amplitude) {
    int totalSamples = audioData.size();
    for (int i = 0; i < totalSamples; ++i) { 
        float carrier = amplitude * sinf(2.0f * M_PI * frequency * i / SAMPLE_RATE);
        audioData[i] *= carrier; 
    } 
}

void AudioFile::TurnMeIntoRobot(){
    this->modulateWithSineWave(100,0.6);
}

AudioFile* ConcatAudio(AudioFile* audio1, AudioFile* audio2){
    // if(audio1->FRAMES_PER_BUFFER != audio2->FRAMES_PER_BUFFER || audio1->NUM_CHANNELS != audio2->NUM_CHANNELS || audio1->SAMPLE_RATE != audio2->SAMPLE_RATE){
    //     printf("Currenlty Only Similar Header Sound-merger has been implemented, please provide apt files\n");
    //     exit(EXIT_FAILURE);
    // }
    if(audio1->NUM_CHANNELS != audio2->NUM_CHANNELS){
        printf("Channels of the two audio should be same for thier merger, convert to same channel first\n");
    }
    std::vector<float> newData;
    newData.reserve(audio1->audioData.size()+audio2->audioData.size());
    newData.insert(newData.end(), audio1->audioData.begin(), audio1->audioData.end());
    newData.insert(newData.end(), audio2->audioData.begin(), audio2->audioData.end());
    AudioFile* mergedAudio = new AudioFile(audio1->NUM_CHANNELS, audio1->SAMPLE_RATE, audio1->FRAMES_PER_BUFFER);
    mergedAudio->audioData = newData;
    return mergedAudio;
}

std::vector<float> CreateSavitzkyGolayKernel(int windowSize){
    if (windowSize % 2 == 0) windowSize++;
    std::vector<float> kernel(windowSize);
    int n = windowSize;
    int m = (n - 1) / 2;
    float norm = 1.0f / (2 * m * (2 * m * m + 1) / 3);
    for (int i = -m; i <= m; i++) {
        float x = i;
        kernel[i + m] = norm * (3 * (m * m - x * x));
    }
    
    return kernel;
}

std::vector<float> CreateMovingAverageKernel(int windowSize) {
    if (windowSize % 2 == 0) windowSize++;
    std::vector<float> kernel(windowSize, 1.0f / windowSize);
    return kernel;
}

std::vector<float> CreateGaussianKernel(int windowSize, float sigma) {
    if (windowSize % 2 == 0) windowSize++;
    std::vector<float> kernel(windowSize);
    int center = windowSize / 2;
    float sum = 0.0f;
    for (int i = 0; i < windowSize; i++) {
        float x = i - center;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < windowSize; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

