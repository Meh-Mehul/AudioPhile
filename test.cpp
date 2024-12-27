#include <stdlib.h>
#include <stdio.h>
#include <portaudio.h>
#include <cstring>
#include <vector>
#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 512

static void checkErr(PaError err){
    if(err != paNoError){
        printf("Port Aduio Error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}
int main(){
    // We do this to safely connect to Audio Devices
    PaError err;
    err = Pa_Initialize();
    checkErr(err);
    int numDevices = Pa_GetDeviceCount();
    printf("Num of devices: %d\n", numDevices);
    if(numDevices < 0 ){
        printf("Error Getting Device Count\n");
        exit(EXIT_FAILURE);
    }
    else if(numDevices == 0){
        printf("No Devices Connected");
        exit(EXIT_SUCCESS);
    }
    std::vector<PaDeviceIndex> id;
    const PaDeviceInfo* deviceInfo;
    for(int i = 0; i<numDevices; ++i){
        deviceInfo = Pa_GetDeviceInfo(i);
        printf("Device %d:\n", i);
        printf("    name: %s\n", deviceInfo->name);
        printf("    MaxIPChannels: %d\n", deviceInfo->maxInputChannels);
        printf("    MaxOPChannels: %d\n", deviceInfo->maxOutputChannels);
        printf("    DefaultSR: %f\n", deviceInfo->defaultSampleRate);
        if(deviceInfo->maxInputChannels > 0 && deviceInfo->maxOutputChannels >0){
            id.push_back(i);
        }
    }
    if(id.size() == 0){
        printf("Could not detect any compatible Drivers.\n");
        exit(EXIT_FAILURE);
    }
    printf("Possible Device Ids:\n");
    for(auto i: id){
        printf(" %d ", i);
    }
    printf("\n");
    // // Safely Terminate the connection
    err = Pa_Terminate();
    checkErr(err);
    return EXIT_SUCCESS;
}