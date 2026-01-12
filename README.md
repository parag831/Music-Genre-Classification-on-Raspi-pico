================================================================================
PROJECT: TINYML MUSIC GENRE CLASSIFICATION ON RP2040
================================================================================

OVERVIEW
--------------------------------------------------------------------------------
This project implements an end-to-end embedded Machine Learning pipeline capable 
of classifying music genres (e.g., Rock, Jazz, Disco) in real-time on a 
Raspberry Pi Pico (RP2040). The system runs entirely on the edge, performing 
Digital Signal Processing (DSP) and Quantized Neural Network inference on a 
Cortex-M0+ microcontroller with only 264KB of RAM.

DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
1. /micsetup & /build_dataset    [Part 1: Hardware & Data Acquisition]
2. /model_training               [Part 2: Deep Learning & Quantization]
3. /music_genre_classification_1 [Part 3: Embedded Firmware]


================================================================================
PART 1: HARDWARE SETUP & DATASET GENERATION (~20%)
Folders: /micsetup, /build_dataset
================================================================================

[micsetup]
This directory contains the initial hardware tests to validate the microphone 
signal chain. The RP2040 ADC (Analog-to-Digital Converter) is configured to 
sample audio at 22,050 Hz.
- DC Bias Calibration: The firmware calculates the zero-point energy of the 
  microphone to remove DC offset.
- Signal Conditioning: Bitwise shifting (<<1) is applied to amplify the 12-bit 
  ADC hardware signal into a full-range 16-bit signed integer format, maximizing 
  dynamic range before processing.

[build_dataset]
Since standard datasets (like GTZAN) do not reflect the specific noise profile 
and frequency response of the cheap electret microphone used in hardware, a 
custom dataset builder was created.
- The script captures raw PCM audio streams from the Pico over USB Serial.
- It slices the audio into standardized 1-second clips.
- This ensures "Data-Centric AI" principles: the training data exactly matches 
  the inference data statistics.


================================================================================
PART 2: MODEL ARCHITECTURE & TRAINING PIPELINE (~60%)
Folder: /model_training
================================================================================

This is the core of the project. It handles Feature Extraction, Model Design, 
Training, and Post-Training Quantization (PTQ).

A. FEATURE EXTRACTION (DSP FRONT-END)
-------------------------------------
Instead of feeding raw audio to the neural network (which requires massive RAM), 
we extract Mel-Frequency Cepstral Coefficients (MFCCs).
1. Framing: Signal sliced into 50ms windows with 50% overlap.
2. Windowing: Hanning window applied to reduce spectral leakage.
3. FFT & Mel-Scale: Converted to frequency domain and mapped to 32 Mel-bands 
   (logarithmic scale mimicking human hearing).
4. DCT: Discrete Cosine Transform applied to decorrelate filterbank energies, 
   resulting in 18 MFCCs per time step.

B. NEURAL NETWORK ARCHITECTURE (LSTM + FUNNEL)
----------------------------------------------
The model is designed specifically for the memory constraints of the RP2040.

1. Stacked LSTM (Long Short-Term Memory):
   - Layer 1 (128 Units, return_seq=True): Extracts temporal rhythmic features.
   - Layer 2 (64 Units, return_seq=False): Condenses the time-series into a 
     single "Concept Vector."
   - Optimization: `unroll=True` is used. This flattens the LSTM loop in the 
     computation graph, increasing Flash usage (code size) but significantly 
     reducing CPU cycles (speed) by removing loop overhead.

2. Dense "Funnel" Classifier:
   - Architecture: Dense(64) -> Dense(32) -> Dense(16) -> Softmax.
   - Design Rationale: Dense layers are "Memory Bound" (requiring unique weight 
     fetches for every calc). This tapered "Funnel" design reduces the parameter 
     count by ~50% compared to a flat structure, minimizing CPU stalls caused 
     by Flash memory latency.

3. Activations:
   - ReLU is used for hidden layers (computationally cheap).
   - Softmax is used only at the final layer for probability distribution.

C. TRAINING STRATEGY
--------------------
- Optimizer: Adam (LR=0.001) for adaptive learning on noisy audio data.
- Loss Function: Sparse Categorical Crossentropy.
- Regularization: Dropout (0.5) and Early Stopping (Patience=10) with 
  `restore_best_weights=True` to prevent overfitting.
- Normalization Export: The training script calculates the Mean and Std-Dev of 
  the dataset and exports them to C++ headers. This guarantees the microcontroller 
  performs the exact same Z-Score normalization as the Python model.

D. QUANTIZATION & COMPILATION
-----------------------------
To fit the model onto the microcontroller, we perform Post-Training Quantization.
1. Concrete Function: The input shape is frozen to [1, 20, 18] to prevent 
   dynamic memory allocation (malloc) errors on the device.
2. INT8 Quantization: Weights and activations are converted from Float32 (4 bytes) 
   to Int8 (1 byte), reducing model size by 4x.
3. Calibration: A `representative_dataset` is run through the converter to 
   determine the dynamic range (min/max) for accurate integer mapping.
4. Header Generation:
   - The TFLite model is dumped as a C byte array.
   - `alignas(8)` is injected into the header. This is critical for the ARM 
     Cortex-M0+, which throws Hard Faults if 64-bit data types are accessed 
     at unaligned memory addresses.


================================================================================
PART 3: EMBEDDED FIRMWARE & INFERENCE (~20%)
Folder: /music_genre_classification_1
================================================================================

This directory contains the C++ firmware utilizing the Pico SDK, TensorFlow Lite 
Micro, and CMSIS-DSP.

1. Real-Time DSP Implementation:
   - The Python MFCC pipeline is replicated bit-for-bit in C++ using ARM's 
     CMSIS-DSP library.
   - Matrices (DCT/Mel) are stored in Flash memory using `const` to save RAM.
   - DSP tables are pre-transposed to optimize for memory access patterns during 
     dot-product calculations.

2. The Inference Loop:
   - Audio is captured via DMA/ISR into a ring buffer.
   - The "Normalization Bridge" applies the pre-calculated (x - mean)/std 
     transformation to the live MFCCs.
   - The TFLite Micro Interpreter (`tflu_interpreter->Invoke()`) executes the 
     quantized model.
   - Output probabilities are smoothed using an Exponential Moving Average (EMA) 
     filter to prevent jittery predictions.

3. Hardware Constraints Handled:
   - Input/Output tensors are kept as Float32 (with internal Int8 execution) to 
     simplify the C++ API, allowing direct injection of DSP results into the model.
================================================================================



**References**

* **[TinyML Cookbook by Gian Marco Iodice]** – DSP optimization & feature extraction.
* **[Musical Genre Classification of Audio Signals](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)** – Seminal paper establishing the GTZAN dataset.
* **[ARM CMSIS-DSP](https://arm-software.github.io/CMSIS-DSP/main/index.html)** – FFT & MFCCs libraries.
* **[IEEE: Music Genre Classification]([https://ieeexplore.ieee.org/document/8941656](https://ieeexplore.ieee.org/document/10605044))** – LSTM architecture research.
* **[Google AI Edge (LiteRT)](https://ai.google.dev/edge/litert/conversion/tensorflow/convert_tf)** – TensorFlow to TFLite quantization.
* **[ProjectPro: Drive Uploads](https://www.projectpro.io/recipes/upload-files-to-google-drive-using-python)** – Cloud automation script.
