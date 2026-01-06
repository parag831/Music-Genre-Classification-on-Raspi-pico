#include <Arduino.h>
#include "hardware/adc.h"
#include "pico/stdlib.h"

const int buttonPin = 10;
#define PRESSED LOW

const int ledPin = LED_BUILTIN;
#define ON HIGH
#define OFF LOW

#define SAMPLE_RATE 22050
#define AUDIO_LENGTH_SEC 4
#define AUDIO_LENGTH_SAMPLES (SAMPLE_RATE * AUDIO_LENGTH_SEC)

struct Buffer {
  volatile int32_t cur_idx = 0;
  volatile bool is_ready = false;
  int16_t data[AUDIO_LENGTH_SAMPLES];
};

volatile Buffer buffer;

struct repeating_timer audio_timer;

#define BIAS_MIC 1552

bool timer_callback(struct repeating_timer *t) {
  if (buffer.cur_idx < AUDIO_LENGTH_SAMPLES) {
    uint16_t raw_val = adc_read();
    int16_t v = (int16_t)(raw_val - BIAS_MIC);
    buffer.data[buffer.cur_idx] = v;
    buffer.cur_idx++;
    return true;
  }
  else {
    buffer.is_ready = true;
    return false;
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);

  adc_init();
  adc_gpio_init(26);
  adc_select_input(0);
}

void loop() {
  if (digitalRead(buttonPin) == PRESSED) {
    delay(800);

    buffer.cur_idx = 0;
    buffer.is_ready = false;

    digitalWrite(ledPin, ON);

    long interval_us = -1000000 / SAMPLE_RATE;
    add_repeating_timer_us(interval_us, timer_callback, NULL, &audio_timer);

    while (!buffer.is_ready) {
      delay(1);
    }

    digitalWrite(ledPin, OFF);

    Serial.println(SAMPLE_RATE);
    Serial.println(AUDIO_LENGTH_SAMPLES);

    for (int i = 0; i < AUDIO_LENGTH_SAMPLES; ++i) {
      Serial.println(buffer.data[i]);
    }

    while (digitalRead(buttonPin) == PRESSED);
  }
}
