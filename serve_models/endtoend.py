import sys
sys.path.append('../code/')
import utils as utls
import argparse
import audiofile as af
import os
import torch
import numpy as np


def separate_from_model(model_str,input_filename):

    if model_str== 'u_net_5_5':
        kernel_size=(5,5)
    if model_str== 'u_net_3_7':
        kernel_size=(7,3)
    if model_str== 'u_net_4_6':
        kernel_size=(6,4)
    audio_mix, fs = af.read(input_filename)
    if (model_str == 'u_net_5_5' or model_str == 'u_net_3_7' or model_str == 'u_net_4_6'):
        model = utls.load_unet_spleeter(kernel_size,'weights/' + model_str + '.hdf5')
        
        if fs!=44100:
            print("Audio must be 44.1 kHz. Exiting")
        audio_vocal_pred = utls.separate_from_audio(audio_mix,44100,model,wiener_filter=True)
        audio_acc_pred = audio_mix - audio_vocal_pred[0:len(audio_mix)]
        
    else:
        audio_prepared = torch.Tensor(audio_mix)
        result = utls.separate_umx(
            audio_prepared, rate=44100, model_str_or_path="weights/model5", targets=['vocals'], residual=True)
        audio_vocal_pred = np.array(result['vocals'][0][0])
        audio_acc_pred = audio_mix[:len(audio_vocal_pred)] - audio_vocal_pred[:len(audio_mix)]

    base_folder = './audio/'
    basename = os.path.splitext(os.path.basename(input_filename))[0]
    af.write(base_folder + basename + '_vocals_pred.wav', audio_vocal_pred, 44100)
    af.write(base_folder + basename + '_acc_pred.wav',audio_acc_pred,44100)
