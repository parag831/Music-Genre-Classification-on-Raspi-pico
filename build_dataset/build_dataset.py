import serial
import soundfile as sf
import numpy as np
import uuid
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

PORT = 'COM6' 
BAUD_RATE = 115200
DEFAULT_LABEL = "extra"
DEFAULT_DRIVE_FOLDER_ID = '1uevyaOMgazZFIgCpC1gc312qaVTUoKJE' 

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

try:
    ser = serial.Serial(PORT, BAUD_RATE)
    ser.reset_input_buffer()
    print(f"Connected to {PORT}")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

def serial_readline(obj):
    try:
        data = obj.readline()
        return data.decode("utf-8").strip()
    except:
        return "0"

print("Ready! Waiting for microcontroller data...")

gdrive_id = DEFAULT_DRIVE_FOLDER_ID

while True:
    try:
        ad_sr_str = serial_readline(ser)
        if not ad_sr_str.isdigit(): 
            continue

        ad_len_str = serial_readline(ser)
        
        ad_sr = int(ad_sr_str)
        ad_len = int(ad_len_str)

        print(f"Receiving {ad_len} samples at {ad_sr} Hz...")

        ad_buf = np.empty((ad_len), dtype=np.int16)
        
        for i in range(0, ad_len):
            sample_str = serial_readline(ser)
            try:
                ad_buf[i] = int(sample_str)
            except ValueError:
                ad_buf[i] = 0

        print("Data received.")

        key = input("Save audio? [y] for YES, any other key to discard: ")

        if key.lower() == 'y':
            str_label = "Provide label name [{}]: ".format(DEFAULT_LABEL)
            label_new = input(str_label)
            
            current_label = label_new if label_new != '' else DEFAULT_LABEL

            unique_id = str(uuid.uuid4())[:8]
            filename = f"{current_label}_{unique_id}.wav"
            
            sf.write(filename, ad_buf, ad_sr, subtype='PCM_16')
            print(f"Saved locally as: {filename}")

            str_gid = "Provide Drive Folder ID [{}]: ".format(gdrive_id)
            gdrive_id_new = input(str_gid)
            
            if gdrive_id_new != '':
                gdrive_id = gdrive_id_new

            print("Uploading to Drive...")
            try:
                gfile = drive.CreateFile({
                    'title': filename,
                    'parents': [{'id': gdrive_id}]
                })
                gfile.SetContentFile(filename)
                gfile.Upload()
                print("SUCCESS: Uploaded to Google Drive!")
            except Exception as e:
                print(f"ERROR: Could not upload to Drive. {e}")
        else:
            print("Discarded.")

        ser.reset_input_buffer()

    except KeyboardInterrupt:
        print("\nStopping program...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

ser.close()