const float MEAN_VAL[] = { 0.930336, 0.340996, -0.201562, 0.215760, -0.121096, 0.140952, -0.129331, 0.125998, -0.109903, 0.100059, -0.091461, 0.079145, -0.072507, 0.057927, -0.063069, 0.050491, -0.050005, 0.035941 };
const float STD_VAL[]  = { 5.618516, 2.427764, 1.374558, 1.279455, 0.891927, 0.878244, 0.798173, 0.770649, 0.699364, 0.647881, 0.604817, 0.547320, 0.528831, 0.463492, 0.466007, 0.420266, 0.409548, 0.372736 };

#include "arm_math.h"
#include "mbed.h"
#include "hardware/adc.h"

// TensorFlow Lite model
#include "model.h"

// MFCCs
#include "mfccs_consts.h"
#include "dct_wei_mtx_q15_T.h"
#include "hann_lut_q15.h"
#include "log_lut_q13_3.h"
#include "mel_wei_mtx_q15_T.h"

// TensorFlow Lite for Microcontrollers
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#define BIAS_MIC    dynamic_bias // (1.25V * 4095) / 3.3
#define SAMPLE_RATE 22050
#define AUDIO_LENGTH_SEC  1
#define AUDIO_LENGTH_SAMPLES (SAMPLE_RATE * AUDIO_LENGTH_SEC)

int16_t dynamic_bias = 0;

#define EXPECTED_INPUT_PEAK  500 

// Calculate the multiplier automatically (32767 is max for int16)
const float NORM_MULTIPLIER = 32767.0f / EXPECTED_INPUT_PEAK;

static const char *label[] = {"disco", "jazz", "metal"};

// TensorFlow Lite for Microcontroller global variables
const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;
constexpr int t_sz                         = 140*1024;
uint8_t tensor_arena[t_sz] __attribute__((aligned(16)));

struct Buffer
{
  int32_t cur_idx{0};
  bool    is_ready{false};
  int16_t data[AUDIO_LENGTH_SAMPLES];
};

mbed::Ticker    timer;
volatile Buffer buffer;

void tflu_initialization() {

  // Load the TFLITE model
  tflu_model = tflite::GetModel(model_tflite);
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while(1);
  }

  static tflite::AllOpsResolver tflu_ops_resolver;

  // Initialize the TFLu interpreter
  static tflite::MicroInterpreter static_interpreter(
        tflu_model,
        tflu_ops_resolver,
        tensor_arena,
        t_sz);

  tflu_interpreter = &static_interpreter;

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  Serial.println("TFLu initialization - completed");
}

class MFCC_Q15 {
public:
 MFCC_Q15() {
    // RFFT instance
    arm_rfft_init_q15(&_rfft_inst, FFT_LENGTH, 0, 1);

    // Mel-weight matrix instance
    _mel_wei_mtx_inst.numRows = mel_wei_mtx_q15_T_dim1;
    _mel_wei_mtx_inst.numCols = mel_wei_mtx_q15_T_dim0;
    _mel_wei_mtx_inst.pData = (q15_t *)&mel_wei_mtx_q15_T_data[0];

    // DCT-weight matrix instance
    _dct_wei_mtx_inst.numRows = dct_wei_mtx_q15_T_dim1;
    _dct_wei_mtx_inst.numCols = dct_wei_mtx_q15_T_dim0;
    _dct_wei_mtx_inst.pData = (q15_t *)&dct_wei_mtx_q15_T_data[0];
  }

  void run(const q15_t* src, float* dst) {
    for(int i = 0; i < NUM_FRAMES; ++i ) {
      // Apply the Hann window
      arm_mult_q15((q15_t*)&src[i * FRAME_STEP], (q15_t*)hann_lut_q15_data, _bufA, FRAME_LENGTH);

      // Calculate the RFFT
      arm_rfft_q15(&_rfft_inst, _bufA, _bufB);

      // Calculate the magnitude
      _bufA[0]                 = _bufB[0];
      _bufA[NUM_FFT_FREQS - 1] = _bufB[1];
      arm_cmplx_mag_q15(&_bufB[2], &_bufA[1], NUM_FFT_FREQS - 2);

      // Mel-scale conversion
      arm_mat_vec_mult_q15(&_mel_wei_mtx_inst, _bufA, _bufB);

      for(int idx = 0; idx < NUM_MEL_FREQS; ++idx) {
        const int16_t val = (int16_t)_bufB[idx];
        _bufA[idx] = log_lut_q13_3_data[val];
      }

      // Calculate the MFCCs through the DCT
      arm_mat_vec_mult_q15(&_dct_wei_mtx_inst, _bufA, _bufB);

      for(int k = 0; k < NUM_MFCCS; ++k) {
        dst[k + i * NUM_MFCCS] = (float)_bufB[k] / (float)(8.0f);
      }
    }
  }
private:
  arm_rfft_instance_q15   _rfft_inst;
  arm_matrix_instance_q15 _mel_wei_mtx_inst;
  arm_matrix_instance_q15 _dct_wei_mtx_inst;
  q15_t                   _bufA[FRAME_LENGTH];
  q15_t                   _bufB[FRAME_LENGTH * 2];
};

MFCC_Q15 mfccs;

void print_raw_audio() {
  Serial.println(SAMPLE_RATE);
  Serial.println(AUDIO_LENGTH_SAMPLES);
  for(int i = 0; i < AUDIO_LENGTH_SAMPLES; ++i) {
    Serial.println((int32_t)buffer.data[i]);
  }
}

void timer_ISR() {
  if(buffer.cur_idx < AUDIO_LENGTH_SAMPLES) {
    int16_t v = (int16_t)((adc_read() - BIAS_MIC));
    // Get current buffer index

    /*if(abs(v)<40)
    {
      v=0;
      }*/
    int32_t ix_buffer = buffer.cur_idx;

    int32_t amp = (int32_t)(v<<1);

       if(amp > 32767)
       {
          amp = 32767;
       }
       if(amp<-32767)
       {
          amp = -32767;
       }
    
    buffer.data[ix_buffer] = (int16_t)amp;
    // Store the sample in the audio buffer
    //buffer.data[ix_buffer] = (int16_t)v;
    // Increment buffer index
    buffer.cur_idx++;
  }
  else {
    buffer.is_ready = true;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

 
  tflu_initialization();

  // Initialize ADC
  adc_init();
  adc_gpio_init(26);
  adc_select_input(0);
  Serial.println("Calibrating... SHH! Be quiet for 2 seconds.");
  long sum = 0;
  for(int i=0; i<5000; i++) {
     sum += adc_read();
     delayMicroseconds(200); 
  }

  dynamic_bias = sum / 5000;
  
  Serial.print("✅ Calibration Complete. Bias found: ");
  Serial.println(dynamic_bias);


Serial.print("Tensor arena size needed: ");
Serial.println(tflu_interpreter->arena_used_bytes());
Serial.print("Tensor arena size allocated: ");
Serial.println(t_sz);
}

void loop() {
  // Reset audio buffer
  buffer.cur_idx  = 0;
  buffer.is_ready = false;

  constexpr uint32_t sr_us = 1000000 / SAMPLE_RATE;
  timer.attach_us(&timer_ISR, sr_us);

  while(!buffer.is_ready);

  timer.detach();

  float energy = 0;
  for (int i = 0; i < AUDIO_LENGTH_SAMPLES; i++) {
      energy += abs(buffer.data[i]);
  }
  energy /= AUDIO_LENGTH_SAMPLES; // Average volume

  // Debug Print: Check your room's "Silence Level"
/* Serial.print("Volume Level: ");
  Serial.println(energy);

  // THRESHOLD: Adjust this number! 
  // If your "Volume Level" prints ~50 when quiet, set this to 100.
  if (energy < 20) { 
      Serial.println("Prediction: SILENCE (Too quiet)");
      return; // Skip the heavy AI processing
  }*/



  // MFCCs computation
  mfccs.run((const q15_t*)&buffer.data[0],
            (float *)&tflu_i_tensor->data.f[0]);
  

  //////////////////////////added
  int total_inputs = tflu_i_tensor->dims->data[1] * tflu_i_tensor->dims->data[2];

// In loop(), just before the normalization loop
float raw_debug = tflu_i_tensor->data.f[0]; // Look at the first feature
Serial.print("Raw MFCC Output: ");
Serial.print(raw_debug);
Serial.print(" | Target Mean: ");
Serial.println(MEAN_VAL[0]); // We want 'raw_debug' to be close to this value (0.93)

  
  for (int i = 0; i < total_inputs; i++) {
      // Get the raw value calculated by mfccs.run
      float raw_val = tflu_i_tensor->data.f[i];

      // Apply Normalization: (Value - Mean) / Std
      // We use (i % 18) because the Mean/Std arrays repeat every 18 features (NUM_MFCCS)
      tflu_i_tensor->data.f[i] = (raw_val - MEAN_VAL[i % 18]) / STD_VAL[i % 18];
  }
  /////////////////////////added
  
  // Run inference
  TfLiteStatus invoke_status = tflu_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the TFLu interpreter");
    return;
  }

  Serial.print("RAW MODEL OUTPUT -> ");
for (size_t ix = 0; ix < 3; ix++) {
    Serial.print(label[ix]);
    Serial.print(": ");
    Serial.print(tflu_o_tensor->data.f[ix], 3);  // 3 decimal places
    Serial.print(" | ");
}
Serial.println();
  size_t ix_max = 0;
  float  pb_max = 0;

  static float smooth_probs[3] = {0, 0, 0};
  float alpha = 0.8;

  for (size_t ix = 0; ix < 3; ix++) {
      smooth_probs[ix] = (alpha * tflu_o_tensor->data.f[ix]) + ((1.0 - alpha) * smooth_probs[ix]);
  }

  for (size_t ix = 0; ix < 3; ix++) {
    if(smooth_probs[ix] > pb_max) {
      ix_max = ix;
      pb_max = smooth_probs[ix];
    }
  }

  Serial.print("Prediction: ");
  Serial.print(label[ix_max]);
  Serial.print(" (Conf: ");
  Serial.print(pb_max);
  Serial.println(")");
  
  /*for (size_t ix = 0; ix < 3; ix++) {
    if(tflu_o_tensor->data.f[ix] > pb_max) {
      ix_max = ix;
      pb_max = tflu_o_tensor->data.f[ix];
    }
  }

  Serial.println(label[ix_max]);*/
}
