# TODO#1 - Import Libraries and Check availability of GPU
import os
import datetime
import whisper
from torch import cuda, Generator

if cuda.is_available():
    Generator('cuda').manual_seed(42)
    print("CUDA available, using GPU.")
else:
    Generator().manual_seed(42)
    print("CUDA not available, using CPU.")

def transcribe(file_path, model=None, language=None, verbose=False):    
    try:
        # TODO#2 - Import Libraries and Check availabity of GPU
        # Load model and output configuration
        model_name = "base"  # You can modify this to "small", "medium", etc.
        model = whisper.load_model(model_name)

        # Output model configuration
        print(f"Model '{model_name}' loaded successfully.")
        print(f"Model dimensions: {model.dims}")

        # TODO#3 - Establish Access to Audio File
        title = os.path.basename(file_path).split('.')[0]
        folder_path = os.path.dirname(file_path)

        print(f"File: {title}")
        print(f"Folder: {folder_path}")
        
        # TODO#4 - Perform transcription and store to `result`
        result = model.transcribe(file_path, language="en", verbose=True)
        print(f"Transcription successful for {title}.")

        # TODO#5 - Create folder where the transcription will be saved to
        # Create folder if missing
        transcription_folder = os.path.join(folder_path, 'transcriptions')
        os.makedirs(transcription_folder, exist_ok=True)

        # TODO#6 - Loop through the result segments.
        # Add start and end time stamps 
        # and save the transcription to a text file
        start = []
        end = []
        text = []
        for segment in result['segments']:
            start.append(str(datetime.timedelta(seconds=segment['start'])))
            end.append(str(datetime.timedelta(seconds=segment['end'])))
            text.append(segment['text'])
                
        with open(os.path.join(transcription_folder, f"{title}.txt"), 'w', encoding='utf-8') as file: file.write(title)
        for i in range(len(result['segments'])):
            file.write(f'\n[{start[i]} --> {end[i]}]:{text[i]}')
            
        print(f"Transcription saved in {transcription_folder}")

    except RuntimeError:
        print("Not a valid file, skipping.")

#Call the transcribe() function:
file_path = 'sample_audio/audio.wav'
transcribe(file_path, model="base", language="en", verbose=True)