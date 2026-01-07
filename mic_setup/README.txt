## Description
This Arduino Sketch captures high-fidelity audio data and streams it to the PC for dataset creation. It is designed to work seamlessly with the
`build_dataset.py` Python script to build a custom-labeled dataset compatible with GTZAN.

{   requires Raspberry Pi Pico/RP2040 by Earle F. Philhower, III board core 
    https://github.com/earlephilhower/arduino-pico.git }
### Operational Logic
1.  **Trigger:** Recording starts when the push button is pressed.

2.  **Debounce/Noise Filter:** A **500ms delay** is introduced immediately after the button
    press to prevent mechanical click sounds from contaminating the audio sample.

3.  **Visual Feedback:** The Onboard LED turns **ON** to indicate recording is active and turns 
    **OFF** after 4 seconds to signal completion.

4.  **Sampling Constraints:**
    * **Duration:** 4 Seconds
    * **Sample Rate:** 22,050 Hz
    * **Memory Usage:** ~176 KB (88,200 samples × 2 bytes). 
This fits within the RP2040's 264 KB SRAM while maintaining 16-bit audio depth.

### Signal Processing
* **Timer Interrupts:** Sampling is driven by a hardware alarm triggering every **~45µs** to ensure zero
jitter and a stable 22.05 kHz sample rate.

* **DC Offset Removal:** A static bias of **-1552** is subtracted from the raw ADC values to
center the waveform around 0.
* **Data Dump:** Once the buffer is full, the main loop pauses recording and dumps the
raw integer array over UART (115200 baud) to the host Python script.

### Connections Required
**Microphone (MAX9814 / MAX4466)**
* `VCC`  →  `3V3` (Pin 36)
* `GND`  →  `GND` (Pin 38)
* `OUT`  →  `GP26` (Pin 31) **[ADC0]**
* `GAIN` →  `3V3` (Pin 36) *(Optional: Sets Gain to 40dB on MAX9814)*

**Push Button**
* `Side A` → `GP10` (Pin 14)
* `Side B` → `GND` (Pin 13)
