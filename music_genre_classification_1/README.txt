================================================================================
FIRMWARE DOCUMENTATION: RP2040 REAL-TIME AUDIO INFERENCE ENGINE
================================================================================

PROJECT: TinyML Music Genre Classification
TARGET:  Raspberry Pi Pico (RP2040) / ARM Cortex-M0+
FRAMEWORK: TensorFlow Lite for Microcontrollers (TFLM) + CMSIS-DSP
VERSION: 1.0.0

--------------------------------------------------------------------------------
1. SYSTEM ARCHITECTURE OVERVIEW
--------------------------------------------------------------------------------
This firmware implements a complete "Edge AI" pipeline consisting of three stages:
1. Data Acquisition (DAQ): Interrupt-driven audio sampling via ADC.
2. Signal Processing (DSP): MFCC extraction using ARM CMSIS-DSP optimizations.
3. Inference Engine (ML): Quantized LSTM execution via TFLite Micro.

The system operates in a "Stop-and-Wait" architecture:
[IDLE] -> [FILL BUFFER (ISR)] -> [DSP + INFERENCE (Main Loop)] -> [OUTPUT]

--------------------------------------------------------------------------------
2. DATA ACQUISITION & INTERRUPT HANDLING
--------------------------------------------------------------------------------
Source: hardware/adc.h (RP2040 Native ADC)
Sample Rate: 22,050 Hz
Bit Depth: 12-bit (Hardware) scaled to 16-bit (Software)

CRITICAL IMPLEMENTATION DETAILS:
- Timer ISR (`timer_ISR`): Triggered every 45.35µs (1/22050Hz).
- Zero-Copy Buffer: The ISR writes directly to `buffer.data[]` to avoid memcpy overhead.
- DC Bias Correction:
  - On startup, `setup()` runs a 5000-sample calibration loop to find the DC floor.
  - `dynamic_bias` is subtracted in real-time inside the ISR.
- Signal Amplification:
  - `v << 1`: Bitwise left shift effectively applies a digital gain of 2x (6dB) 
    to utilize the full signed 16-bit dynamic range (-32768 to 32767).
  - Hard clipping is applied at +/- 32767 to prevent integer overflow wrapping.

--------------------------------------------------------------------------------
3. DIGITAL SIGNAL PROCESSING (DSP) PIPELINE
--------------------------------------------------------------------------------
Class: `MFCC_Q15`
Library: ARM CMSIS-DSP (Fixed-Point Q15 Arithmetic)

The DSP pipeline transforms raw Time-Domain PCM into Frequency-Domain MFCC features.
It leverages the RP2040's SIO hardware multiplier where possible via CMSIS.

PIPELINE STAGES:
A. Windowing (`arm_mult_q15`):
   - Multiplies input frame by `hann_lut_q15` (Hanning Window).
   - Mitigates spectral leakage caused by non-periodic frame boundaries.

B. FFT (`arm_rfft_q15`):
   - Performs Real-Fast Fourier Transform.
   - Converts 320 time samples -> 256 frequency bins (Complex).
   - Magnitude calculated via `arm_cmplx_mag_q15`.

C. Mel-Filterbank (`arm_mat_vec_mult_q15`):
   - Matrix Multiplication: [32 x 256] Matrix * [256 x 1] Vector.
   - Maps linear FFT frequency bins to 32 Mel-scale bands (human hearing model).
   - Uses `mel_wei_mtx_q15_T` (Transposed for memory access locality).

D. Logarithm (`log_lut_q13_3`):
   - Lookup Table based log calculation (Log2 or Ln approximation).
   - Converts Linear Energy to Logarithmic Energy (Decibels).
   - Crucial for dynamic range compression.

E. DCT (`arm_mat_vec_mult_q15`):
   - Discrete Cosine Transform via matrix multiplication.
   - Decorrelates Mel-energies into 18 Cepstral Coefficients (MFCCs).

--------------------------------------------------------------------------------
4. TENSORFLOW LITE MICRO (TFLM) INITIALIZATION
--------------------------------------------------------------------------------
The model runs on a static memory arena to prevent heap fragmentation.

A. MODEL LOADING (`tflite::GetModel`):
   - Reads the `model_tflite` byte array from `model.h`.
   - Validates `TFLITE_SCHEMA_VERSION`. Mismatches between the training schema 
     and TFLM library schema will trigger a halt here.

B. TENSOR ARENA (`tensor_arena`):
   - Size: 140KB (Defined by `t_sz`).
   - Alignment: `__attribute__((aligned(16)))`.
   - Purpose: Stores Input/Output tensors AND intermediate activation buffers (scratch memory) 
     required by the LSTM states and Dense layers during inference.

C. INTERPRETER (`tflite::MicroInterpreter`):
   - Static allocation. No `malloc()` is used during runtime.
   - `AllocateTensors()`: Maps the computation graph onto the `tensor_arena` at startup.

--------------------------------------------------------------------------------
5. THE NORMALIZATION BRIDGE (CRITICAL)
--------------------------------------------------------------------------------
The model was trained on Z-Score normalized data (Mean=0, Std=1), but the DSP 
pipeline outputs raw energy values. A "Bridge" loop is implemented in `loop()` 
before inference.

Implementation:
   float raw_val = tflu_i_tensor->data.f[i];
   tflu_i_tensor->data.f[i] = (raw_val - MEAN_VAL[i%18]) / STD_VAL[i%18];

- `MEAN_VAL[]` and `STD_VAL[]`: Constant arrays exported from the Python training script.
- Modulo Operator (`i % 18`): Applies the correct stats to each of the 18 MFCCs 
  across all time steps (Broadcast normalization).

--------------------------------------------------------------------------------
6. INFERENCE & POST-PROCESSING
--------------------------------------------------------------------------------
A. EXECUTION:
   - `tflu_interpreter->Invoke()`: Triggers the TFLite compute graph.
   - Runs the Quantized LSTM -> Dense Layer stack.
   - Returns Softmax probabilities in `tflu_o_tensor`.

B. OUTPUT SMOOTHING (EMA Filter):
   - Raw neural net outputs can jitter between frames.
   - An Exponential Moving Average (EMA) filter is applied:
     P_smooth = (alpha * P_current) + ((1 - alpha) * P_prev)
   - Alpha: 0.8 (Prioritizes new data while smoothing transient noise).

--------------------------------------------------------------------------------
7. MEMORY MAP SUMMARY
--------------------------------------------------------------------------------
[FLASH] 2MB Total
  - model_data[]:   ~30KB (Quantized Model Weights)
  - hann_lut[]:     ~640B (DSP LUT)
  - mel_mtx[]:      ~16KB (Filterbank Weights)
  - code text:      Remainder

[RAM] 264KB Total
  - tensor_arena:   140KB (Model Runtime)
  - buffer.data:    44KB  (1 sec Audio Ring Buffer)
  - Stack/Heap:     Remainder

================================================================================
END OF DOCUMENTATION
================================================================================
