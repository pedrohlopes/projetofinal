from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import subprocess
import os
from endtoend import separate_from_model

app = Flask(__name__,static_url_path='')


@app.route('/upload/')
def upload_file():
    return render_template('upload.html')
    
@app.route('/upload/uploader/', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        print('debug')
        f = request.files['file']
        wr_filename =secure_filename(f.filename)
        f.save(wr_filename)
        print('Salvei o arquivo local: ', wr_filename)
        wav_filename = os.path.splitext(wr_filename)[0] + "_converted" + ".wav"
        bashCommand = ["ffmpeg", "-i", wr_filename, "-ar", "44100", "-ac", "1", wav_filename]
        process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
        process.wait()
        print('Converti o arquivo')
        model = request.form.get('select_model')
        separate_from_model(model,wav_filename)
        print('Separei')
        basename = os.path.splitext(os.path.basename(wav_filename))[0]
        print(basename)
        os.remove(wr_filename)
        os.remove(wav_filename)
        final_filename_vocals = secure_filename(basename + "_vocals_pred.wav")
        final_filename_acc = secure_filename(basename + "_acc_pred.wav")
        print(final_filename_vocals,final_filename_acc)
        return render_template('download.html', f_vocals=final_filename_vocals,f_acc=final_filename_acc)


@app.route('/upload/downloader/', methods= ['GET', 'POST'])
def download_file():
    
    if request.method == 'POST':
        filename = request.form['part']
        return send_file('audio/' + filename, as_attachment=True)
                                    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
