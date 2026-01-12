Step 1: Framing (Slicing the Audio)Audio changes constantly.
Analyzing a whole second at once is too messy.
Action: We slice the continuous audio signal into small, overlapping chunks called Frames.
Typical Size:50ms per frame for 20 frames each having 50% overlap with adjacent frames to preserve timing 
sequence of our data.
Why: In these tiny windows, the sound is "stationary" (constant). It captures a single instant of a vowel 
or consonant.

Step 2: Windowing (Smoothing Edges)The Problem: When you slice audio, you create sharp "cliffs" at 
the start and end of the frame. The math treats these sharp cuts as high-frequency noise 
("spectral leakage").The Solution: We multiply the frame by a Hanning Window 
(the bell curve we discussed).Result: The audio fades in from zero and fades out to zero, smoothing the 
edges.Code Connection: This uses the hann_lut_q15 table.

Step 3: FFT (Fast Fourier Transform)Action: We convert the signal from the Time Domain 
(Amplitude vs. Time) to the Frequency Domain (Energy vs. Frequency).The Output: 
A "Periodogram" or Spectrum. It tells us: "How much bass is here? How much treble?"Why: To identify 
a word, we don't care about the wave shape; we care about the frequencies (formants) that make up the 
sound.

Step 4: Mel Filterbank (The "Human Ear" Step): The FFT gives us linear details 
(e.g., 100Hz bins). But human ears are logarithmic. We can distinguish 100Hz from 200Hz easily, 
but 10,000Hz and 10,100Hz sound exactly the same to us.
to solve this We apply a bank of triangular filters (usually 20-40 of them) spaced according to 
the Mel Scale.
Low frequencies: Narrow filters (lots of detail).
High frequencies: Wide filters (less detail, grouped together).
Result: We sum up the energy in each filter. If we had 256 FFT points, we might condense them down 
to just 32 Mel Bands.Code Connection: This uses mel_wei_mtx_q15 matrix.

Step 5: Logarithm (The "Decibel" Step): Humans hear loudness logarithmically. A sound that 
is $100\times$ more powerful physically only sounds $2\times$ louder to us.Action: We take 
the Logarithm of the energy in each Mel band: $Log(Energy)$.Why: This compresses the dynamic 
range. It ensures that a shout doesn't mathematically "drown out" the subtle features of the sound,
and it makes the features match how humans perceive loudness (Decibels): 
This is done through log_lut_q13 table.

Step 6: DCT (Discrete Cosine Transform)This is the final "cleaning" step that makes MFCCs special.
The Problem: The Mel filterbank energies are highly correlated. (If Filter 1 is high energy, 
Filter 2 is usually high energy too because sound "bleeds" across bands). Neural Networks hate 
correlated input; they want distinct, independent features.Action: We apply the DCT. It separates 
the Spectral Envelope (the overall shape of the sound, like the shape of your mouth) from the fine 
pitch details.Result: We keep only the first few coefficients (e.g., 10 to 13). These are 
the MFCCs.Code Connection: This is done through dct_wei_mtx_q15 matrix.
