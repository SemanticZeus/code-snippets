#include <ofApp.h>

float phase[7];
float frequencies[7];
float amplitude = 5;
float sampleRate = 44100;

void audioOut(ofSoundBuffer &buffer) {
    for (size_t i = 0; i < buffer.getNumFrames(); i++) {
        float sample = 0.0;
        
        for (int j = 0; j < 7; j++) {
            sample += amplitude * sin(phase[j]) / 7.0; // Normalize the sum
            phase[j] += TWO_PI * frequencies[j] / sampleRate;
            if (phase[j] > TWO_PI) {
                phase[j] -= TWO_PI;
            }
        }
        
        buffer[i * buffer.getNumChannels()] = sample;        // Left channel
        buffer[i * buffer.getNumChannels() + 1] = sample;    // Right channel
    }
}
