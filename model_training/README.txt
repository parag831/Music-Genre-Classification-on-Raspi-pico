# model.py – Audio Feature Extraction & LSTM Architecture

This document describes the design decisions, signal-processing pipeline,
training strategy, and embedded deployment constraints for `model.py`.

This file is responsible for:
- MFCC feature extraction
- LSTM-based sequence modeling
- Training, validation, and quantization
- Preparing the model for RP2040 deployment

> ⚠️ Important:
> Any change to MFCC parameters (frame size, Mel bins, DCT size)
> must be reflected in both `model.py` and the embedded preprocessing code.
> Mismatches will cause silent inference failures.

## PreProcessing

### Step 1: Framing (Slicing the Audio)Audio changes constantly.
Analyzing a whole second at once is too messy.
Action: We slice the continuous audio signal into small, overlapping chunks called Frames.
Typical Size:50ms per frame for 20 frames each having 50% overlap with adjacent frames to preserve timing 
sequence of our data.
Why: In these tiny windows, the sound is "stationary" (constant). It captures a single instant of a vowel 
or consonant.

### Step 2: Windowing (Smoothing Edges)The Problem: When you slice audio, you create sharp "cliffs" at 
the start and end of the frame. The math treats these sharp cuts as high-frequency noise 
("spectral leakage").The Solution: We multiply the frame by a Hanning Window 
(the bell curve we discussed).Result: The audio fades in from zero and fades out to zero, smoothing the 
edges.Code Connection: This uses the hann_lut_q15 table.

### Step 3: FFT (Fast Fourier Transform)Action: We convert the signal from the Time Domain 
(Amplitude vs. Time) to the Frequency Domain (Energy vs. Frequency).The Output: 
A "Periodogram" or Spectrum. It tells us: "How much bass is here? How much treble?"Why: To identify 
a word, we don't care about the wave shape; we care about the frequencies (formants) that make up the 
sound.

### Step 4: Mel Filterbank (The "Human Ear" Step): The FFT gives us linear details 
(e.g., 100Hz bins). But human ears are logarithmic. We can distinguish 100Hz from 200Hz easily, 
but 10,000Hz and 10,100Hz sound exactly the same to us.
to solve this We apply a bank of triangular filters (usually 20-40 of them) spaced according to 
the Mel Scale.
Low frequencies: Narrow filters (lots of detail).
High frequencies: Wide filters (less detail, grouped together).
Result: We sum up the energy in each filter. If we had 256 FFT points, we might condense them down 
to just 32 Mel Bands.Code Connection: This uses mel_wei_mtx_q15 matrix.

### Step 5: Logarithm (The "Decibel" Step): Humans hear loudness logarithmically. A sound that 
is 100 times more powerful physically only sounds 2 times louder to us.Action: We take 
the Logarithm of the energy in each Mel band: Log(Energy).Why: This compresses the dynamic 
range. It ensures that a shout doesn't mathematically "drown out" the subtle features of the sound,
and it makes the features match how humans perceive loudness (Decibels): 
This is done through log_lut_q13 table.

### Step 6: DCT (Discrete Cosine Transform)This is the final "cleaning" step that makes MFCCs special.
The Problem: The Mel filterbank energies are highly correlated. (If Filter 1 is high energy, 
Filter 2 is usually high energy too because sound "bleeds" across bands). Neural Networks hate 
correlated input; they want distinct, independent features.Action: We apply the DCT. It separates 
the Spectral Envelope (the overall shape of the sound, like the shape of your mouth) from the fine 
pitch details.Result: We keep only the first few coefficients (e.g., 10 to 13). These are 
the MFCCs.Code Connection: This is done through dct_wei_mtx_q15 matrix.

In non-embedded environments, Mel spectrograms are often preferred due to their higher frequency resolution. However, for TinyML and embedded deployment, MFCCs are chosen for two key reasons: reduced model size and faster convergence.However, for TinyML and Embedded , we stick to MFCCs for two specific reasons: Model Size and Convergence Speed.


## Model (Audio → MFCC → LSTM(128) → LSTM(64) → Dense Funnel → Softmax → Genre)

Model Summary

- Input: 20 time steps × 18 MFCC coefficients
- Feature extraction: MFCC (Hann → FFT → Mel → Log → DCT)
- Sequence model: LSTM (128 → 64)
- Classifier: Dense funnel (64 → 32 → 16 → Softmax)
- Deployment target: RP2040 (TFLite Micro, INT8)



### Normalization is performed manually because, on embedded devices, it is significantly more efficient than adding a dedicated normalization layer.


### LSTM layer stack
The Model that we have used uses a simple architecture with a 128 , 64 stack LSTM layer ,LSTMs (Long Short-Term Memory networks) are preferred because they solve the fundamental problem of "Time" and "Memory". Unlike standard RNNs, LSTMs solve the Vanishing Gradient problem using a Cell State and Gating mechanisms (Forget, Input, Output). This allows the model to learn long-term dependencies, effectively 'remembering' the start of the word 'Alexa' or a note of song while processing the end of it. LSTMs are highly parameter-efficient for streaming data, making them ideal for the constrained memory of a microcontroller. 

Layer 1: The "Sequence Preserver" (return_sequences=True)
Input: A sequence of audio features (e.g., 50 time steps of MFCCs).

Job: This layer analyzes the audio step-by-step. However, instead of giving one final summary at the end, it outputs a new sequence of the same length.

Output: For every single time step input, it outputs a vector of size 128.

Why: This allows the second LSTM layer to see the "history" of the sequence. If Layer 1 only outputted a summary, Layer 2 would lose the ability to see how the song evolves over time. It essentially "translates" raw audio features into "abstract musical features" while keeping the time dimension intact.

Layer 2: The "Summarizer" (return_sequences=False)
Input: The sequence of 128-sized vectors from Layer 1.

Job: It reads through the entire translated sequence, updating its internal state.

Output: Once it reaches the very last time step, it discards the sequence and outputs only its final internal state (a vector of size 64).

Why: This collapses the Time Dimension ,we  are converting a 1-second song clip into a single static "Concept Vector" (Embedding) that represents the essence of that clip. This static vector is what the Dense layers (the classifier) need to make a prediction.

The Hardware Optimization: unroll=True
When unroll=True, TensorFlow "flattens" the loop. It generates a massive chain of identical operations

The Cost: This increases Program Memory (Flash) usage significantly because the code is physically longer.

The Benefit: It executes much faster because there is zero loop management overhead.

Why for RP2040? The RP2040 has 2MB of Flash (huge for a micro) but a relatively slow Cortex-M0+ core ,So we are trading your abundant Flash storage to for raw speed.

The "Funnel" Sizing (128 to 64 LSTM): { Why this bottleneck is introduced instead of using 128 to 128 LSTM stack}

Layer 1 (128 Units): This is "Wide Net." It needs a large capacity to capture every possible detail in the raw audio (bass kicks, hi-hats, vocal pitch, tempo).

Layer 2 (64 Units): This forces the model to prioritize. It cannot keep all 128 features; it has to condense them into the 64 most important patterns.

The Result: By the time data leaves Layer 2, noise has been filtered out, and only the strong, defining characteristics of the genre remain.

After the LSTM stack layer a Dropout layer with 50% droprate is used to prevent overfitting as LSTMs can easily overfit when trained on limited audio data.

### The Dense Layer Architecture: 

Instead of using one big layer,We used a tapered structure that gets smaller at every step. This is a deliberate design choice often called a Bottleneck or Funnel Architecture.

Dense(64): Interpretation

Input: Receives the 64-value "Concept Vector" from the LSTM.

Role: This layer tries to make sense of the LSTM's output. It looks for combinations of features. For example, it might combine "Fast Tempo" (from LSTM) + "Distorted Guitar" (from LSTM) to create a new internal feature "High Energy."

Dense(32) & Dense(16): DistillationRole: These layers force the model to compress its knowledge. By reducing the number of neurons (32 to 16), you force the network to throw away irrelevant details and keep only the strongest, most essential evidence.
Why for RP2040? This drastically reduces the parameter count. A Dense(64->64) layer has 4,160 parameters. A Dense(64->32) layer has only 2,080 , thereby saving 50% of the Flash memory for that layer while maintaining high accuracy.

In terms of hardware execution on the RP2040, a Dense layer is the simplest but most memory-intensive operation.The Operation: Output = ReLU((Input × Weights) + Bias). This is pure Matrix Multiplication.For the Dense(64) layer taking 64 inputs from the LSTM:The RP2040 must perform 64 * 64 = 4,096 multiply-accumulate (MAC) operations.It must fetch 4,096 distinct weights from Flash memory.
Memory Bandwidth: Unlike Convolutional layers (which reuse weights), Dense layers use a weight only once per inference. This makes them "Memory Bound" i.e. CPU wastes more time waiting for data then actual execution. The funnel design helps here by keeping the matrices small, preventing the CPU from stalling while waiting for Flash memory.

After this the dense layer introduces ReLu Activation so that we only get the data that is non zero and relavent for our use.

The Final Layer: Softmax
The last layer is unique: Dense(len(LIST_GENRES), activation='softmax').

The Role: Probability Distribution.

The Math: It takes the raw numbers (Logits) from the previous layer (e.g., [2.5, 0.1, 4.8]) and squashes them so they all sum up to exactly 1.0 (100%).

Input: [Rock: 5, Jazz: 2, Pop: 1]

Output: [Rock: 0.95, Jazz: 0.04, Pop: 0.01]


## Learning and Validation

### The Optimizer (Adam):
Adam (Adaptive Moment Estimation) is the algorithm that adjusts the weights.

Unlike standard Stochastic Gradient Descent (SGD) which uses a fixed step size, Adam adapts the learning rate for each individual weight. If a weight is oscillating wildly (uncertainty), Adam slows it down. If a weight is moving steadily, Adam speeds it up. This is crucial for audio data, which can be noisy.

### Learning Rate (0.001): This is the "Step Size." Too high, and the model overshoots the minimum (fails to converge). Too low, and training takes forever. 0.001 is the industry standard "safe" starting point.

The Loss Function (Sparse Categorical Crossentropy): The labels (y_train) are integers (e.g., 0 for Jazz, 1 for Rock).

### Early Stopping

This is the most critical line for preventing Overfitting.

Neural Networks are essentially "memorization machines." If you train for too long (e.g., 100 epochs), the model will start memorizing the specific static noise in your training files instead of learning general musical patterns.

This callback watches the val_loss (Validation Loss).

If the training loss goes down (model learning) but validation loss goes up (model memorizing), it means overfitting has started.

patience=10: It waits 10 epochs to make sure it's not just a random blip.

restore_best_weights=True:  Even if the model overfits at epoch 45 and stops at 55, this automatically rewinds the model back to epoch 45 (the peak performance) before saving.

###  The Training Loop (.fit)

This executes the standard Backpropagation cycle:

Forward Pass: The model takes a batch of 32 audio clips (BATCH_SIZE=32), runs them through the LSTM, and guesses the genre.

Loss Calculation: It compares the guess to the actual label using the Loss Function.

Backward Pass: It calculates the gradient (error) and sends it backward through the network.

Weight Update: The Adam optimizer nudges the weights slightly to reduce the error for next time.

Validation Check: At the end of every epoch (one full pass through the dataset), it pauses training and tests itself on X_val (data it has never seen) to calculate the "real" accuracy.

## Evaluation Metrics.

### Accuracy: The percentage of correct guesses.

### F1-Score: The harmonic mean of Precision and Recall.
Why it matters: If your dataset has 90 Rock songs and 10 Jazz songs, a "dumb" model could just guess "Rock" every time and get 90% accuracy. The F1-score would be terrible (near 0 for Jazz), instantly revealing that the model is biased.

### Confusion Matrix: This is a grid that shows exactly where the model is failing.
If conf_mat[0, 1] is high, it means the model frequently thinks "Jazz" (0) is actually "Rock" (1). This hints at similarity in the data (maybe your Jazz clips are too aggressive?).

## The Embedded "Handoff"

print(f"float MEAN_VAL[] = {{ {mean_str} }}");
print(f"float STD_VAL[]  = {{ {std_str} }}");
This is the bridge for Normalization layer on microcontroller.

Context: The model.fit() process learned weights based on normalized data (inputs strictly between -2 and +2 roughly).
If you deploy this to the microcontroller and feed it raw audio (values like 2000, -500), the LSTM will output garbage because the inputs are outside the "expected range."
By printing these values here, you guarantee that the C++ code on the microcontroller performs the exact same mathematical transformation ((x - mean) / std) as the Python code did during training.


## Post Training Quantization and Deployement onto the microcontroller

### Freezing the Graph (get_concrete_function)

run_model = tf.function(lambda x: model(x, training=False))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(shape=[1, 20, 18], dtype=tf.float32)
)
TensorFlow models are usually "dynamic." They can handle batch sizes of 1, 32, or 100 on the fly. Microcontrollers hate dynamic memory allocation; they need to know exactly how much RAM to reserve at compile time. get_concrete_function traces the model and "freezes" the input shape to [1, 20, 18] (Batch Size 1, 20 Time Steps, 18 MFCCs). The converter generates a static computation graph. The microcontroller code will now hard-code the memory buffers, preventing "Out of Memory" errors during inference.

converter.optimizations = [tf.lite.Optimize.DEFAULT]
This enables Quantization. It prepares the converter to switch from 32-bit floats (4 bytes per number) to 8-bit integers (1 byte per number), effectively shrinking the model size by 4x.

### The Calibration Step (representative_dataset)
converter.representative_dataset = representative_data_gen

To convert a float like 0.573 to an integer like 42, the converter needs to know the Range (Min and Max) of every single neuron in the network.

If a layer's output is always between 0 and 1, int8 has high precision.

If it's between -1000 and +1000, int8 has low precision.

The converter runs a few hundred real audio samples (your representative_dataset) through the model. It records the min/max values of every layer activation.  It calculates the precise "Scale" and "Zero-point" parameters needed to map the float world to the integer world.


### The LSTM Special Case (_experimental_lower_tensor_list_ops)

converter._experimental_lower_tensor_list_ops = False

This is a crucial "magic flag" for LSTMs.

LSTMs use "TensorLists" to store the state over time steps. By default, TensorFlow tries to break these down into tiny primitive operations ("lowering"). Setting this to False tells the converter: "Don't break the LSTM apart. Keep it as a high-level LSTM operation."

Why? The TensorFlow Lite Micro library has a highly optimized, hand-written kernel for the LSTM block. If you let the converter break it apart, you lose that optimization and the model might not run at all on the microcontroller.

### The alignas(8) for model: 
!sed -i 's/const/alignas(8) const/g' model.h

The RP2040 (ARM Cortex-M0+) fetches memory in 32-bit (4-byte) chunks. However, the TFLite Micro library often tries to cast pointers to int64_t or double for certain internal math operations.
If your model array starts at memory address 1001 (an odd number) and the processor tries to read a 64-bit value (8 bytes) starting there, the CPU will physically choke. It cannot read across that boundary in one goThe microcontroller throws a Hard Fault (the screen freezes, or it reboots).

The Fix: alignas(8) forces the C++ compiler to store your model array at a memory address that is perfectly divisible by 8 (e.g., 2000 or 2008). This guarantees that every data fetch inside the model is perfectly aligned with the hardware bus.

## The PreProcessing Header's

### to_c_array():This function is a custom script designed to serialize NumPy arrays into C syntax.

The code generates four distinct "Mathematical Tools" for the microcontroller.
###hann_lut_q15 (The Window)
Input: hann_lut_q15 (NumPy Array).
Output: hann_lut_q15.h.
Content: The pre-calculated curve values (0 to 32767 to 0).
Usage: Used to multiply the audio buffer to prevent spectral leakage.

### mel_wei_mtx_q15_T (The Filterbank)
Input: mel_wei_mtx_q15.T (Notice the .T).Why the .T (Transpose)?
Memory Access Optimization: In the matrix multiplication loop on the microcontroller, we want to read memory sequentially (Address 1, 2, 3) rather than jumping around (Address 1, 50, 99). Transposing the matrix in Python often aligns the data with the specific dot-product algorithm used in the CMSIS-DSP library, making the math run faster.

### log_lut_q13_3 (The Logarithm)Input: log_lut_q13_3.Content: A lookup table that maps Energy values to Log-Energy values.Calculating log() on a CPU without an FPU is slow. Looking up a value in an array is instant.

### dct_wei_mtx_q15_T (The Compressor)Input: dct_wei_mtx_q15.T.Content: The Discrete Cosine Transform weights.Used to decorrelate the Mel-features and compress them into MFCCs.
