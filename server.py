import os, sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent path to sys.path to import se_extractor from ../se_extractor

import torch
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter

ckpt_base = '../checkpoints/base_speakers/EN'
ckpt_converter = '../checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'
output_se_dir = f'{output_dir}/se'
output_generated_dir = f'{output_dir}/generated'

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_se_dir, exist_ok=True)
os.makedirs(output_generated_dir, exist_ok=True)

# Maybe change this in the future
source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

# create a HTTP endpoint to accept mp3 files to make new speaker

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import hashlib

app = Flask(__name__)
CORS(app)

@app.route('/add_voice', methods=['POST'])
# Input: mp3 file and unique speaker name
# Output: Success or Failure
def add_voice():
    try:
        speaker_name = request.form['speaker_name']
        # If speaker already exists (i.e. speaker_name.mp3 already exists), return failure
        if os.path.exists(f'{output_dir}/{speaker_name}.mp3'):
            return jsonify({'status': 'failure'})

        mp3_file = request.files['voice']
        mp3_file.save(f'{output_dir}/{speaker_name}.mp3')

        # Extract speaker embedding
        reference_speaker = f'{output_dir}/{speaker_name}.mp3'
        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir=output_se_dir, vad=True)

        return jsonify({'status': 'success'})
    except Exception as e:
        print(e)
        return jsonify({'status': 'failure'})

@app.route('/synthesize', methods=['POST'])
# Input: text, speaker name
# Output: synthesized mp3 file name
def synthesize():
    try:
        text = request.form['text']
        speaker_name = request.form['speaker_name']

        # If speaker doesn't exist (i.e. speaker_name.mp3 doesn't exist) or text is empty, return failure
        if not os.path.exists(f'{output_dir}/{speaker_name}.mp3') or not text:
            return jsonify({'status': 'failure'})

        # If speaker exists, load the speaker embedding
        reference_speaker = f'{output_dir}/{speaker_name}.mp3'
        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir=output_se_dir, vad=True)

        # Synthesize
        # file name is speaker_name-sha256(text).wav
        generated_file_name = f'{speaker_name}-{hashlib.sha256(text.encode()).hexdigest()}.wav'
        save_path = f'{output_generated_dir}/{generated_file_name}'

        # if the file doesn't exist, synthesize
        if not os.path.exists(save_path):
            # Run the base speaker TTS
            src_path = f'{output_dir}/tmp.wav'
            base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)

            # Convert the tone and color
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=save_path,
                message=""
            )

        # send the file name to the client
        return jsonify({'status': 'success', 'file_name': generated_file_name})        
        
    except Exception as e:
        print(e)
        return jsonify({'status': 'failure'})
    

# Get endpoint to get synthesized mp3 file
# Input file name
# Output: synthesized mp3 file
@app.route('/get_file', methods=['GET'])
def get_file():
    try:
        file_name = request.args.get('file_name')
        return send_file(f'{output_generated_dir}/{file_name}', mimetype='audio/wav', as_attachment=True, attachment_filename=file_name)
    except Exception as e:
        print(e)
        return jsonify({'status': 'failure'})
    
# Echo endpoint to test if the server is running
@app.route('/echo', methods=['POST'])
# Input: text
# Output: text
def echo():
    try:
        text = request.form['text']
        return jsonify({'status': 'success', 'text': text})
    except Exception as e:
        print(e)
        return jsonify({'status': 'failure'})
    
# Hello world get route on the root path
@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'


# run on port 80 so that it can be accessed from outside
app.run(host="0.0.0.0", port=80, debug=True)