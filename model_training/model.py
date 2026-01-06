!pip install tensorflow-model-optimization
import tensorflow_model_optimization as tfmot

!pip install numpy==1.23.5
!pip install cmsisdsp==1.9.9

SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
FRAME_STEP = 1024
FFT_LENGTH = 2048
FMIN_HZ = 20
FMAX_HZ = SAMPLE_RATE / 2
NUM_MEL_FREQS = 40
NUM_MFCCS = 18

from google.colab import drive
import soundfile as sf
drive.mount('/content/drive/')

train_dir = "/content/drive/My Drive/mgr_dataset"

file_path = train_dir+'/disco/disco.00002.wav'

ad,sr = sf.read(file_path)

SAMPLE_RATE = 22050
test_ad = ad[0:SAMPLE_RATE]

import numpy as np
import tensorflow as tf
import IPython.display as ipd

# Redefine the function with the fix for tf.matmul and dct indexing
def extract_mfccs_tf(
    ad_src,
    ad_sample_rate,
    num_mfccs,
    frame_length,
    frame_step,
    fft_length,
    fmin_hz,
    fmax_hz,
    num_mel_freqs
    ):

  n=ad_src.shape[0]
  num_frames=int((n-frame_length)/frame_step)
  num_frames +=int(1)

  output = np.zeros(shape=(num_frames,num_mfccs))

  hann_coeff=tf.signal.hann_window(frame_length)

  for i in range(num_frames):
    idx_s = i * frame_step
    idx_e = idx_s + frame_length
    src = ad_src[idx_s:idx_e]
    hann=src * hann_coeff
    fft_spect=tf.signal.rfft(hann)
    fft_mag_spect = tf.math.abs(fft_spect)
    num_fft_freqs = fft_mag_spect.shape[0]
    mel_wei_mtx = tf.signal.linear_to_mel_weight_matrix(num_mel_freqs,num_fft_freqs,ad_sample_rate,fmin_hz,fmax_hz)
    # Fix 1: Expand dimensions of fft_mag_spect to make it a row vector for tf.matmul
    mel_spect = np.matmul(tf.expand_dims(fft_mag_spect, 0), mel_wei_mtx)
    # Fix 2: Use tf.math.log instead of np.log for TensorFlow tensors
    log_mel_spect = tf.math.log(mel_spect + 1e-6)
    dct = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spect)
    # Fix 3: Correct indexing for dct to get the first num_mfccs coefficients from the single frame
    output[i] = dct[0, 0:num_mfccs]
  return output

ipd.Audio(test_ad, rate=sr)

mfccs_tf = extract_mfccs_tf(
test_ad,
SAMPLE_RATE,
NUM_MFCCS,
FRAME_LENGTH,
FRAME_STEP,
FFT_LENGTH,
FMIN_HZ,
FMAX_HZ,
NUM_MEL_FREQS)

import matplotlib.pyplot as plt
from matplotlib import cm
def display_mfccs(mfccs_src):
  fig, ax = plt.subplots()
  cax = ax.imshow(mfccs_src, interpolation='nearest', cmap=cm.gray,
  origin='lower')
  ax.set_title('MFCCs')
  plt.xlabel('Frame index - Time')
  plt.ylabel('Coefficient index - Frequency')
  plt.colorbar(cax)
  plt.show()

display_mfccs(mfccs_tf.T)

import matplotlib.pyplot as plt
from matplotlib import cm

def display_mfccs(mfcc_src):
  fig, ax = plt.subplots()
  cax = ax.imshow(mfcc_src, interpolation='nearest', cmap=cm.gray, origin='lower')
  ax.set_title('MFCCs')
  plt.xlabel('Frame index - Time')
  plt.ylabel('Coefficient index - Frequency')
  plt.colorbar(cax)
  plt.show()

import cmsisdsp as dsp

def rfft_q15(src):
  inst = dsp.arm_rfft_instance_q15()
  src_len=src.shape[0]
  stat = dsp.arm_rfft_init_q15(inst,src_len,0,1)
  fft_q15=dsp.arm_rfft_q15(inst,src)
  return fft_q15[:src_len + 1]



def mag_q15(src):
  f0=src[0],
  fn=src[1],
  fx=dsp.arm_cmplx_mag_q15(src[2:])
  return np.concatenate((f0,fx,fn))


src = test_ad[0:FRAME_LENGTH]
src_q15=dsp.arm_float_to_q15(src)
cmsis_fft_q15=rfft_q15(src_q15)
cmsis_mag_q15=mag_q15(cmsis_fft_q15)
scale = float(1<<3)
cmsis_fft_mag=cmsis_mag_q15/scale
tf_fft=tf.signal.rfft(src)
tf_fft_mag=tf.math.abs(tf_fft)
abs_diff=np.abs(tf_fft_mag-cmsis_fft_mag)
print("DIFF:\n",
"min:", np.min(abs_diff),
"max:", np.max(abs_diff),
"mean:", np.mean(abs_diff),
"std:", np.std(abs_diff))
def gen_hann_lut_q15(frame_len):
  hann_lut_f32 = tf.signal.hann_window(frame_len)
  return dsp.arm_float_to_q15(hann_lut_f32
                              )
def gen_mel_weight_mtx(sr, fmin_hz, fmax_hz,num_mel_freqs, num_fft_freqs):
  m_f32 = tf.signal.linear_to_mel_weight_matrix(num_mel_freqs,num_fft_freqs,sr,fmin_hz,fmax_hz)
  m_q15 = dsp.arm_float_to_q15(m_f32)
  return m_q15.reshape((m_f32.shape[0],m_f32.shape[1]))

def gen_log_lut_q(q_scale):
  max_int16 = np.iinfo("int16").max
  log_lut = np.zeros(shape=(max_int16), dtype="int16")
  for i16 in range(0, max_int16):
    q16 = np.array([i16,], dtype="int16")
    f_v = q16 / float(q_scale)
    log_f = np.array(np.log(f_v + 1e-6),)
    log_q = log_f * float(q_scale)
    log_lut[i16] = int(log_q)
  return log_lut
import math
def gen_dct_weight_mtx(num_mel_freqs, num_mfccs):
  mtx_q15 = np.zeros(shape=(num_mel_freqs, num_mfccs),dtype="int16")
  scale = np.sqrt(2.0 / float(num_mel_freqs))
  pi_div_mel = (math.pi / num_mel_freqs)
  for n in range(num_mel_freqs):
    for k in range(num_mfccs):
      v = scale * np.cos(pi_div_mel * (n + 0.5) * k)
      v_f32 = np.array([v,], dtype="float32")
      mtx_q15[n][k] = dsp.arm_float_to_q15(v_f32)
  return mtx_q15

num_fft_freqs = int((FFT_LENGTH / 2) + 1)
q13_3_scale = 8
# Precompute the Hann window coefficients
hann_lut_q15 = gen_hann_lut_q15(FRAME_LENGTH)
# Precompute the Mel-weight matrix
mel_wei_mtx_q15 = gen_mel_weight_mtx(SAMPLE_RATE,FMIN_HZ,FMAX_HZ,NUM_MEL_FREQS,num_fft_freqs)
# Precompute the Log function for Q13.3
log_lut_q13_3 = gen_log_lut_q(q13_3_scale)
# Precompute the DCT-weight matrix
dct_wei_mtx_q15 = gen_dct_weight_mtx(NUM_MEL_FREQS,NUM_MFCCS)
num_bytes = 2
mem_usage = 0
mem_usage += np.size(hann_lut_q15) * num_bytes
mem_usage += np.size(mel_wei_mtx_q15) * num_bytes
mem_usage += np.size(log_lut_q13_3) * num_bytes
mem_usage += np.size(dct_wei_mtx_q15) * num_bytes
print("Program memory usage: ", mem_usage, "bytes")

def extract_mfccs_cmsis(ad_src,num_mfccs,frame_length,frame_step,hann_lut_q15,mel_wei_mtx_q15,log_lut_q13_3,dct_wei_mtx_q15):
  n = ad_src.shape[0]
  num_frames = int((n - frame_length) / frame_step)
  num_frames += int(1)
  output = np.zeros(shape=(num_frames, num_mfccs))
  for i in range(num_frames):
    idx_s = i * frame_step
    idx_e = idx_s + frame_length
    frame = ad_src[idx_s:idx_e]
    frame_q15 = dsp.arm_float_to_q15(frame)
    frame_q15 = dsp.arm_mult_q15(frame_q15,hann_lut_q15)
    fft_spect_q15 = rfft_q15(frame_q15)
    fft_mag_spect_q15 = mag_q15(fft_spect_q15)
    log_mel_spect_q15 = dsp.arm_mat_vec_mult_q15(mel_wei_mtx_q15.T,fft_mag_spect_q15.T)
    for idx, v in enumerate(log_mel_spect_q15):
      log_mel_spect_q15[idx] = log_lut_q13_3[v]
      mfccs = dsp.arm_mat_vec_mult_q15(dct_wei_mtx_q15.T,log_mel_spect_q15)
      output[i] = mfccs.T / float(8)
    return output

mfccs_cmsis = extract_mfccs_cmsis(test_ad,NUM_MFCCS,FRAME_LENGTH,FRAME_STEP,hann_lut_q15,mel_wei_mtx_q15,log_lut_q13_3,dct_wei_mtx_q15)
display_mfccs(mfccs_cmsis.T)
abs_diff = np.abs(mfccs_tf - mfccs_cmsis)
display_mfccs(abs_diff.T)

import os
x=[]
y=[]
LIST_GENRES =['disco','jazz','metal']
for genre in LIST_GENRES :
  folder = train_dir + '/' + genre
  list_files = os.listdir(folder)
  for song in list_files:
    file_path = folder + '/' + song
    try :
      ad , sr = sf.read(file_path)
      TRAIN_AUDIO_LENGTH_SAMPLE = 22050

      num_it = int(len(ad)/TRAIN_AUDIO_LENGTH_SAMPLE)
      for i in range(num_it) :
        s0 = i * TRAIN_AUDIO_LENGTH_SAMPLE
        s1 = s0 + TRAIN_AUDIO_LENGTH_SAMPLE
        src_audio = ad[s0:s1]
        mfccs = extract_mfccs_cmsis(src_audio,NUM_MFCCS,FRAME_LENGTH,FRAME_STEP,hann_lut_q15,mel_wei_mtx_q15,log_lut_q13_3,dct_wei_mtx_q15)
        x.append(mfccs.tolist())
        y.append(LIST_GENRES.index(genre))

    except Exception as e:
          continue

x,y=np.array(x),np.array(y)

#LSTM32_32
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

X_train, X_0, y_train, y_0 = train_test_split(x, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_0, y_0, test_size=0.5, random_state=3)

train_mean = np.mean(X_train, axis=(0, 1))
train_std = np.std(X_train, axis=(0, 1))

train_std = train_std + 1e-6

X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

input_shape = (X_train.shape[1], X_train.shape[2])

model_LSTM32_32 = Sequential()
model_LSTM32_32.add(layers.InputLayer(shape=input_shape))
model_LSTM32_32.add(layers.LSTM(128, return_sequences=True,unroll=True))
model_LSTM32_32.add(layers.LSTM(64,return_sequences=False,unroll=True))
model_LSTM32_32.add(layers.Dropout(0.5))
model_LSTM32_32.add(layers.Dense(units=64, activation='relu'))
model_LSTM32_32.add(layers.Dense(units=32, activation='relu'))
model_LSTM32_32.add(layers.Dense(units=16, activation='relu'))
model_LSTM32_32.add(layers.Dense(len(LIST_GENRES), activation='softmax'))

model_LSTM32_32.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 100
BATCH_SIZE = 32

early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model_LSTM32_32.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1,
    callbacks=[early_stopper]
)

model_LSTM32_32.summary()

y_pred_proba = model_LSTM32_32.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test F1-Score: {f1:.4f}")

print("\n--- Confusion Matrix ---")
print("          Predicted 0   Predicted 1")
print(f"Actual 0: {conf_mat[0, 0]:<12} {conf_mat[0, 1]}")
print(f"Actual 1: {conf_mat[1, 0]:<12} {conf_mat[1, 1]}")

mean_str = ", ".join([f"{x:.6f}" for x in train_mean])
print(f"float MEAN_VAL[] = {{ {mean_str} }}");

print("") # Spacer

# Generate Std Array string
std_str = ", ".join([f"{x:.6f}" for x in train_std])
print(f"float STD_VAL[]  = {{ {std_str} }}");

print(X_train.max())
print(X_train.min())

def representative_data_gen():
  data = tf.data.Dataset.from_tensor_slices(X_test)
  for i_value in data.batch(1).take(100):
    i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
    yield [i_value_f32]

import tensorflow as tf
model = model_LSTM32_32

#run_model = tf.function(lambda x: model(x, training=False))
#concrete_func = run_model.get_concrete_function(
#    tf.constant(X_test[:1].astype(np.float32))  # Use real data
#)



run_model = tf.function(lambda x: model(x,training=False))

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(shape=[1, 20, 18], dtype=tf.float32)
)

print("Concrete function generated:", concrete_func)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=model)

# SAFE FUSION FOR PICO

#converter.target_spec.supported_types = tf.float32  # Or [tf.int8] for quantized
converter.inference_input_type = tf.float32 # Match your input type
converter.inference_output_type = tf.float32  # Match your output type


converter.optimizations = [tf.lite.Optimize.DEFAULT] # <--- This triggers standard fusion

converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # <--- Safe for Pico
converter._experimental_lower_tensor_list_ops = False



#with open("model_dynamic.tflite", "wb") as f:
#    f.write(tflite_model)

tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

print("TFLite Interpreter Loaded.")

# --- STEP 2: Run Inference Loop ---
tflite_predictions = []

print(f"Testing {len(X_test)} samples...")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Keras Base Accuracy: {accuracy * 100:.2f}%")

for i in range(len(X_test)):

    input_sample = X_test[i]
    input_sample = np.expand_dims(input_sample, axis=0)
    input_sample = input_sample.astype(np.float32)
    interpreter.set_tensor(input_index, input_sample)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)
    interpreter.reset_all_variables()
    predicted_class = np.argmax(output_data)
    tflite_predictions.append(predicted_class)

# --- STEP 3: Calculate Accuracy ---
acc = accuracy_score(y_test, tflite_predictions)

print("\n" + "="*30)
print(f"TFLite Accuracy: {acc * 100:.2f}%")
print("="*30)

TFL_MODEL_FILE = 'model.tflite'
with open(TFL_MODEL_FILE, "wb") as f:
    f.write(tflite_model)

!xxd -i $TFL_MODEL_FILE > model.h
!sed -i 's/unsigned char/const unsigned char/g' model.h
!sed -i 's/const/alignas(8) const/g' model.h
def to_c_array(data, c_type, filename, num_cols = 12):

  def to_numpy_dt(dtype):
    if dtype == 'float':
      return 'float32'
    if dtype == 'int32_t':
      return 'int32'
    if dtype == 'uint32_t':
      return 'uint32'
    if dtype == 'int16_t':
      return 'int16'
    if dtype == 'uint16_t':
      return 'uint16'
    if dtype == 'int8_t':
      return 'int8'
    if dtype == 'uint8_t':
      return 'uint8'
    return ''

  str_out = ''

  # Write the header guard
  header_guard = filename.upper()
  str_out += '#ifndef ' + header_guard + '\n'
  str_out += '#define ' + header_guard + '\n'

  # Write the tensor dimensions
  # Scan the dimensions in reverse order
  dim_base = 'const int32_t ' + filename + '_dim'
  for idx, dim in enumerate(data.shape[::-1]):
    str_out += dim_base + str(idx) + ' = '
    str_out += str(dim)
    str_out += ';\n'

  # Reshape the NumPy array and cast the array to desired C data type
  np_type  = to_numpy_dt(c_type)
  data_out = data.flatten()
  data_out = data_out.astype(np_type)

  # Write the tensor total size (Optional)
  size = len(data_out)
  sz_base = 'const int32_t ' + filename + '_sz'
  str_out += sz_base + ' = '
  str_out += str(size) + ';\n'

  # Write the array definition
  str_out += 'const ' + c_type + ' ' + filename + '_data[] = '
  str_out += "\n{\n"

  # Write the values
  for i, val in enumerate(data_out):
    str_out += str(val)

    if (i + 1) < len(data_out):
      str_out += ','
    if (i + 1) % num_cols == 0:
      str_out += '\n'

  str_out += '};\n'
  str_out += '#endif\n'

  # Save the C header file
  h_filename = filename + '.h'
  open(h_filename, "w").write(str_out)

def to_c_consts(data, filename):
  str_out = ''

  # Write the header guard
  header_guard = filename.upper()
  str_out += '#ifndef ' + header_guard + '\n'
  str_out += '#define ' + header_guard + '\n'

  for x in data:
    value    = x[0]
    var_name = x[1]
    c_type   = x[2]
    str_out += 'const ' + c_type + ' '
    str_out += var_name + ' = '
    str_out += str(value)
    str_out += ';\n'

  str_out += '#endif\n'

  # Save the C header file
  h_filename = filename + '.h'
  open(h_filename, "w").write(str_out)

to_c_array(hann_lut_q15, 'int16_t','hann_lut_q15')
to_c_array(mel_wei_mtx_q15.T, 'int16_t','mel_wei_mtx_q15_T')
to_c_array(log_lut_q13_3, 'int16_t','log_lut_q13_3')
to_c_array(dct_wei_mtx_q15.T, 'int16_t','dct_wei_mtx_q15_T')
test_src_q15 = dsp.arm_float_to_q15(test_ad)
to_c_array(test_src_q15, 'int16_t', 'test_src')
to_c_array(mfccs_cmsis, 'float', 'test_dst')
NUM_FRAMES = int(((TRAIN_AUDIO_LENGTH_SAMPLE - FRAME_LENGTH) / FRAME_STEP) + 1)
NUM_FFT_FREQS = int((FFT_LENGTH / 2) + 1)
vars = [(FRAME_LENGTH, 'FRAME_LENGTH', 'int32_t'),(FRAME_STEP, 'FRAME_STEP', 'int32_t'),(NUM_FRAMES, 'NUM_FRAMES', 'int32_t'),(FFT_LENGTH, 'FFT_LENGTH', 'int32_t'),(NUM_FFT_FREQS, 'NUM_FFT_FREQS', 'int32_t'),(NUM_MEL_FREQS, 'NUM_MEL_FREQS', 'int32_t'),(NUM_MFCCS, 'NUM_MFCCS', 'int32_t')]
to_c_consts(vars, 'mfccs_consts')
