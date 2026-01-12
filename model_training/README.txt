================================================================================
DOCUMENTATION: model.py
AUDIO FEATURE EXTRACTION & LSTM ARCHITECTURE DESIGN
================================================================================

OVERVIEW
--------------------------------------------------------------------------------
This document details the design decisions, signal-processing pipeline, training 
strategy, and embedded deployment constraints implemented in `model.py`. 

This script is the "Brain" of the project, responsible for:
1. Digital Signal Processing (MFCC Extraction)
2. Sequence Modeling (LSTM Architecture)
3. Quantization & Compilation for the RP2040 (Cortex-M0+)

> ⚠️ CRITICAL WARNING:
> Any change to DSP parameters (Frame Size, Mel Bins, DCT Coefficients) here 
> must be manually reflected in the C++ firmware. Mismatches will result in 
> silent inference failures where the model receives "alien" data.

================================================================================
1. PREPROCESSING PIPELINE (The "Funnel")
================================================================================
Raw audio is high-dimensional and noisy. We transform it into a compact "fingerprint" 
using Mel-Frequency Cepstral Coefficients (MFCCs). This pipeline is designed to 
run identically in Python (training) and C++ (inference).



STEP 1: FRAMING (Slicing)
- Action: Audio is sliced into overlapping windows.
- Config: 50ms frame length, 50% overlap.
- Why: Audio is non-stationary. We capture "instantaneous" sounds (phonemes/notes) 
  by analyzing small, stationary chunks. Overlap preserves the temporal sequence.

STEP 2: WINDOWING (Smoothing)
- Artifact: Slicing creates sharp "cliffs" at edges, causing high-frequency 
  spectral leakage.
- Solution: Apply a Hanning Window (`hann_lut_q15`) to fade edges to zero.
- Effect: Minimizes noise in the subsequent FFT step.

STEP 3: FFT (Fast Fourier Transform)
- Action: Converts Time Domain (Amplitude) -> Frequency Domain (Energy).
- Output: A Spectrum/Periodogram revealing frequency composition.
- Why: Audio classification relies on formants (pitch/timbre), not wave shape.

STEP 4: MEL FILTERBANK (Human Hearing Simulation)
- Problem: FFT is linear, but human hearing is logarithmic (we distinguish low 
  freqs better than high freqs).
- Solution: Apply a bank of triangular filters spaced on the Mel Scale.
- Config: Compresses ~256 FFT bins into ~32 Mel Bands (`mel_wei_mtx_q15`).
- Hardware Note: We use Mel Matrices instead of Spectrograms to reduce RAM usage 
  on the microcontroller.

STEP 5: LOGARITHM (Dynamic Range Compression)
- Action: Calculate Log(Energy) using `log_lut_q13`.
- Why: Mimics human loudness perception (Decibels). Ensures loud noises don't 
  mathematically drown out subtle spectral details.

STEP 6: DCT (Decorrelation)
- Action: Apply Discrete Cosine Transform (`dct_wei_mtx_q15`).
- Why: Mel bands are highly correlated (sound bleeds across filters). Neural 
  Networks struggle with correlated input. DCT separates the "Spectral Envelope" 
  from fine pitch details.
- Result: We keep the first 18 coefficients (MFCCs) as the final input feature.

================================================================================
2. MODEL ARCHITECTURE (The "Brain")
================================================================================
We utilize a Stacked LSTM followed by a "Funnel" Dense classifier. This architecture 
prioritizes temporal learning while optimizing for the RP2040's limited memory bandwidth.



A. THE LSTM STACK
-----------------
Layer 1: The "Sequence Preserver" (LSTM 128 Units)
- Config: `return_sequences=True`
- Input: 20 Time Steps x 18 MFCCs.
- Role: Translates raw audio features into abstract musical features for *every* time step. It preserves the Time Dimension so the next layer can see the 
  progression of the song.

Layer 2: The "Summarizer" (LSTM 64 Units)
- Config: `return_sequences=False`
- Role: Collapses the Time Dimension. It reads the entire sequence and outputs a 
  single static "Concept Vector" (Embedding) representing the essence of the clip.
- Sizing: Reduced from 128 to 64 to force the model to filter noise and prioritize 
  only the strongest patterns.

> HARDWARE OPTIMIZATION: `unroll=True`
> - Effect: Flattens the LSTM loop into a long chain of explicit operations.
> - Trade-off: Increases Flash usage (Code Size) but drastically reduces CPU 
>   overhead (Execution Speed). 
> - Target: RP2040 has 2MB Flash (Plenty) but slow CPU. This is a favorable trade.

B. THE DENSE "FUNNEL" CLASSIFIER
--------------------------------
Structure: Dense(64) -> Dense(32) -> Dense(16) -> Softmax
- Design: Tapered bottleneck architecture.
- Hardware Reason: Dense layers are "Memory Bound" (Weights loaded once per op). 
  A large single layer stalls the CPU waiting for Flash memory. The funnel reduces 
  parameter count by ~50% compared to a flat design, improving cache coherence.
- Activation: ReLU for hidden layers, Softmax for final probability distribution.

================================================================================
3. TRAINING STRATEGY
================================================================================
- Optimizer: Adam (Learning Rate 0.001). Adapts step size for noisy audio gradients.
- Loss: Sparse Categorical Crossentropy.
- Regularization: 
  1. Dropout (50%) after LSTM to prevent memorization.
  2. Early Stopping (Patience=10): Monitors `val_loss`. If the model starts 
     overfitting (loss increases), training halts. `restore_best_weights=True` 
     ensures we revert to the optimal state.

THE NORMALIZATION BRIDGE
------------------------
The model is trained on normalized data (Mean=0, Std=1). The microcontroller 
sees raw integer data.
- Action: The script calculates the Global Mean and Std Dev of the training set.
- Output: Prints C++ arrays (`MEAN_VAL`, `STD_VAL`) to the console.
- Result: Ensures the firmware performs the exact same math: `(x - mean) / std`.

================================================================================
4. DEPLOYMENT: QUANTIZATION & COMPILATION
================================================================================
To run on the RP2040, the model must be frozen, quantized, and serialized.

A. FREEZING THE GRAPH
- Command: `get_concrete_function` with `tf.TensorSpec`
- Why: Microcontrollers cannot handle dynamic memory allocation. We hard-code 
  the input shape to [1, 20, 18], allowing the compiler to pre-allocate all buffers.

B. INT8 QUANTIZATION
- Technique: Post-Training Quantization (PTQ).
- Action: Converts Float32 weights (4 bytes) to Int8 (1 byte).
- Calibration: Runs a `representative_dataset` through the model to determine 
  dynamic ranges (Min/Max) for accurate scaling factors.

C. LSTM PRESERVATION
- Flag: `converter._experimental_lower_tensor_list_ops = False`
- Critical: Prevents TFLite from breaking LSTMs into primitive ops. This forces 
  the use of the highly-optimized TFLite Micro LSTM kernel.

D. MEMORY ALIGNMENT (The Crash Preventer)
- Command: `sed -i 's/const/alignas(8) const/g' model.h`
- Issue: Cortex-M0+ throws Hard Faults if 64-bit data types are accessed at 
  non-64-bit aligned memory addresses.
- Fix: Forces the C++ compiler to align the model array on an 8-byte boundary.

================================================================================
5. HEADER GENERATION (Helper Scripts)
================================================================================
The firmware cannot import NumPy or Scipy. We use a custom helper function 
`to_c_array()` to serialize Python mathematical constants into C++ headers.

A. HANN WINDOW (`hann_lut_q15.h`)
- Source: NumPy Hanning window generator.
- Format: Fixed-point Q15 (Integers -32768 to 32767 representing -1.0 to 1.0).
- Usage: Applied to the audio buffer inside the ISR/loop to smooth spectral leakage.

B. MEL FILTERBANK (`mel_wei_mtx_q15_T.h`)
- Content: The mapping matrix from FFT bins to Mel bands.
- Optimization (`.T`): The matrix is **Transposed** in Python before export.
- Why: Hardware matrix multiplication reads memory sequentially (Row-Major vs 
  Col-Major). By transposing in Python, we align the data structure with the 
  access pattern of the CMSIS-DSP `arm_mat_vec_mult` function, significantly 
  reducing CPU cycles per inference.

C. LOGARITHM LUT (`log_lut_q13_3.h`)
- Problem: The Cortex-M0+ lacks a Floating Point Unit (FPU). Calculating `log()` 
  via software emulation is prohibitively slow.
- Solution: A Pre-calculated Lookup Table (LUT).
- Operation: Instead of computing `log(x)`, the CPU essentially performs 
  `Array[x]`. This turns a complex math op into a near-instant O(1) memory read.

D. DCT MATRIX (`dct_wei_mtx_q15_T.h`)
- Content: Discrete Cosine Transform weights.
- Role: Decorrelates the Mel-features, compressing them into the final MFCCs.
- Optimization: Also transposed (`.T`) for CMSIS-DSP compatibility.

E. MEMORY PLACEMENT (`const`)
- Critical: All generated arrays are declared as `const`.
- Hardware Implication: This forces the Linker to store these large tables in 
  **Flash Memory** (2MB available), keeping them out of **RAM** (only 264KB available). 
  Removing `const` would immediately cause a Stack Overflow / Out of Memory crash.
