'''
Pre-requisites for running this file:
    1. make a folder named uploads in the same dir as this file
'''
import os
from flask import Flask, request, jsonify #jsonify converts a dict into json
from random import randint
from predict import predict_path
import fleep
from datetime import datetime

# Create a directory in a known location to save the correct files into it.
uploads_dir = os.path.join(os.getcwd(), 'uploads')
#if the sent file isn't in the correct format, it will be saved to rubbish folder not uploads, the purpose of this is to 
uncorrectFiles_dir = os.path.join(os.getcwd(), 'uncorrectFiles') 

def getAudioExtension(audio_path):
    with open(audio_path, "rb") as file:
        info = fleep.get(file.read(128))

    #return info.type #['audio']
    print("Extension: ", end='')
    print(info.extension)
    return info.extension # ['wav']
    #return info.mime #['audio/wav']

def saveRequestFile():

    if len(request.files) == 0 and len(request.form) == 0:
        return "fail message: empty request"
    elif not("baby_cry_record" in request.files):
        return "fail message: baby_cry_record key isn't found in the request files"

    file_sent = request.files['baby_cry_record'] #read sent file
    if file_sent.filename == '':
        return "fail message: no record sent"
    
    file_sent.save(os.path.join(uploads_dir, str(file_sent.filename))) #save the received file
    if not (os.path.exists(os.path.join(uploads_dir, str(file_sent.filename)))):
        return "fail message: file isn't saved"

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    absolute_filename = os.path.join('uploads', file_sent.filename)
    audio_extension = getAudioExtension(absolute_filename)[0]
    file_new_name = dt_string + "__" + str(randint(1000, 9999)) + '.' + audio_extension #make a random file name for the names conflicts
    
    if (audio_extension != 'wav'):
        new_filename = os.path.join('uncorrectFiles', file_new_name)
        os.rename(absolute_filename, new_filename)
        return "fail message: File Format is not wav, it is: " + audio_extension
    
    new_filename = os.path.join('uploads', file_new_name)
    os.rename(absolute_filename, new_filename)
    
    return file_new_name
    

# Init
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return jsonify({'Connection': 'Succeeded'})
    #return "<html><body><H1>Successful Connection</H1></body></html>"

@app.route('/predict', methods=['POST'])
def upload():
    print("Request Files: ", end='')
    print(request.files)
    print("Request Form: ", end='')
    print(request.form)
    
    file_path = saveRequestFile()
    
    if "fail message: " in file_path:
        return jsonify({'operation': 'FAIL', 'baby_crying_reason': file_path})
    
    crying_reason = predict_path(os.path.join('uploads', file_path))
    return jsonify({'operation': 'SUCCESS', 'baby_crying_reason': crying_reason})

# Run Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)#, use_reloader=False)

