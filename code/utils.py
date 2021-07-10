import scaper
import random
import audiofile as af
from glob import glob
from os import path
import numpy as np
import numpy as np
import keras
import datetime
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
import norbert
from functools import partial
from openunmix import utils
import openunmix
import json


json_path = '../norm_data_full.json'
with open(json_path) as infile:
        norm_data = json.load(infile)
dataset_path = '/home/pedro.lopes/data/audio_data/train/augmented/'
X_mean = norm_data['X_min']
X_std = norm_data['X_max'] - norm_data['X_min']

def preprocess_audio(audio,rate,return_stft=False):
    samples = []
    num_samples = 20
    offset = 1500
    freq_bins = 512
    sample_len = 128
    stft_data = tf.signal.stft(audio, 
	frame_length=1024, 
	frame_step=512,
	fft_length=1024).numpy().T
    


    X_mean = norm_data['X_min']
    X_std = norm_data['X_max'] - norm_data['X_min']
    mod_input = np.zeros((int(sample_len*np.ceil(len(stft_data[0])/sample_len)),513),dtype = complex)
    mod_input[0:len(stft_data[0])]= np.array(stft_data.T)

    
    range_check = range(0,int(sample_len*np.ceil(len(stft_data[0])/sample_len)),sample_len)
    for i in range_check:
        samples.append(mod_input[i:i+sample_len,1:513].T.reshape(freq_bins,sample_len,1))
    mod_input_array = preprocess(np.array(samples),X_mean,X_std)
    if return_stft:
        return mod_input_array,stft_data,samples
    return mod_input_array

def normalize(x,save=False):
    scaled_x = (x - np.mean(x))/(np.abs(np.std(x))+1e-8)

    if save:
      return scaled_x, np.mean(x), np.std(x)
    return scaled_x
def normalize_from_outer(x,x_mean,x_std):
  scaled_x = (x - x_mean)/(x_std+1e-8)
  return scaled_x
def preprocess(sample,x_mean,x_std):
  log_sample = np.log(np.abs(sample)+1e-7)
  mod_input = normalize_from_outer(log_sample,x_mean,x_std)
  return mod_input

def preprocess_tf(sample,x_mean,x_std):
  log_sample = tf.math.log(tf.math.abs(sample)+1e-7)
  mod_input = normalize_from_outer(log_sample,x_mean,x_std)
  return mod_input

def denormalize(x,x_mean,x_std):
  scaled_x = x*(x_std + 1e-8) + x_mean
  return scaled_x


def pad_tf(stft_data,test=False):
    x_before = tf.keras.backend.permute_dimensions(stft_data,(0,2,1))
    x_shape = tf.cast(tf.shape(x_before)[-1],tf.float64)
    pad_len = tf.math.floor(tf.math.ceil(x_shape/128)*128 - x_shape)
    pad = ([0,0],[0,0],[0,pad_len])
    x_padded = tf.pad(x_before,pad,mode='constant', constant_values=0)
    if test:
        x_split = tf.split(x_padded,int(x_padded.shape[-1]/128),axis = -1)
    else:
        x_split = tf.split(x_padded,7,axis = -1)
    x = tf.concat(x_split,axis=0)
    return x,x_padded

def preprocess_audio_tf(l_input,test=False):
    stft_data = tf.signal.stft(l_input, 
    frame_length=1024, 
    frame_step=512,
    fft_length=1024)
    x,x_padded = pad_tf(stft_data,test)
    x_padded = tf.squeeze(x_padded)
    x = preprocess_tf(x,X_mean,X_std)[:,0:512]
    x = tf.expand_dims(x,axis=-1)
    if test==True:
        return x,x_padded
    return x



def incoherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials, an event template, 
    and a random seed, and returns an INCOHERENT mixture (audio + annotations). 
    
    Stems in INCOHERENT mixtures may come from different songs and are not temporally
    aligned.
    
    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will 
        generate different mixtures for the same source material and event template.
        
    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """
    
    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=10.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )
    
    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1
    
    # Copy the template so we can change it
    event_parameters = event_template.copy()
    
    # Iterate over stem types and add INCOHERENT events
    labels = ['vocals', 'acc']
    for label in labels:
        event_parameters['label'] = ('const', label)
        sc.add_event(**event_parameters)
    
    # Return the generated mixture audio + annotations 
    # while ensuring we prevent audio clipping
    return sc.generate(fix_clipping=False)


def coherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials and a random seed,
    and returns an COHERENT mixture (audio + annotations).
    
    Stems in COHERENT mixtures come from the same song and are temporally aligned.
    
    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will 
        generate different mixtures for the same source material and event template.
        
    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """
        
    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=10.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )
    
    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1
    
    # Copy the template so we can change it
    event_parameters = event_template.copy()    
    
    # Instatiate the template once to randomly choose a song,   
    # a start time for the sources, a pitch shift and a time    
    # stretch. These values must remain COHERENT across all stems
    sc.add_event(**event_parameters)
    event = sc._instantiate_event(sc.fg_spec[0])
    
    # Reset the Scaper object's the event specification
    sc.reset_fg_event_spec()
    
    # Replace the distributions for source time, pitch shift and 
    # time stretch with the constant values we just sampled, to  
    # ensure our added events (stems) are coherent.              
    event_parameters['source_time'] = ('const', event.source_time)
    event_parameters['pitch_shift'] = ('const', event.pitch_shift)
    event_parameters['time_stretch'] = ('const', event.time_stretch)

    # Iterate over the four stems (vocals, drums, bass, other) and 
    # add COHERENT events.                                         
    labels = ['vocals', 'acc']
    for label in labels:
        
        # Set the label to the stem we are adding
        event_parameters['label'] = ('const', label)
        
        # To ensure coherent source files (all from the same song), we leverage
        # the fact that all the stems from the same song have the same filename.
        # All we have to do is replace the stem file's parent folder name from "vocals" 
        # to the label we are adding in this iteration of the loop, which will give the 
        # correct path to the stem source file for this current label.
        coherent_source_file = event.source_file.replace('vocals', label)
        event_parameters['source_file'] = ('const', coherent_source_file)
        # Add the event using the modified, COHERENT, event parameters
        sc.add_event(**event_parameters)
    
    # Generate and return the mixture audio, stem audio, and annotations
    return sc.generate(fix_clipping=False)


def get_unet_spleeter(input_tensor, kernel_size=(5, 5), strides=(2, 2)):
    DROPOUT = 0
    conv_activation_layer = LeakyReLU(0.2)
    deconv_activation_layer = ReLU()
    conv_n_filters = [16, 32, 64, 128, 256, 512]
    kernel_initializer = 'he_normal'
    conv2d_factory = partial(
        Conv2D, strides=strides, padding="same", kernel_initializer=kernel_initializer
    )
    # First layer.
    conv1 = conv2d_factory(conv_n_filters[0], kernel_size)(input_tensor)
    batch1 = BatchNormalization(axis=-1)(conv1)
    rel1 = conv_activation_layer(batch1)
    # Second layer.
    conv2 = conv2d_factory(conv_n_filters[1], kernel_size)(rel1)
    batch2 = BatchNormalization(axis=-1)(conv2)
    rel2 = conv_activation_layer(batch2)
    # Third layer.
    conv3 = conv2d_factory(conv_n_filters[2], kernel_size)(rel2)
    batch3 = BatchNormalization(axis=-1)(conv3)
    rel3 = conv_activation_layer(batch3)
    # Fourth layer.
    conv4 = conv2d_factory(conv_n_filters[3], kernel_size)(rel3)
    batch4 = BatchNormalization(axis=-1)(conv4)
    rel4 = conv_activation_layer(batch4)
    # Fifth layer.
    conv5 = conv2d_factory(conv_n_filters[4], kernel_size)(rel4)
    batch5 = BatchNormalization(axis=-1)(conv5)
    rel5 = conv_activation_layer(batch5)
    # Sixth layer
    conv6 = conv2d_factory(conv_n_filters[5], kernel_size)(rel5)
    batch6 = BatchNormalization(axis=-1)(conv6)
    _ = conv_activation_layer(batch6)
    #
    #
    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=strides,
        padding="same",
        kernel_initializer=kernel_initializer,
    )
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], kernel_size)((conv6))
    up1 = deconv_activation_layer(up1)
    batch7 = BatchNormalization(axis=-1)(up1)
    drop1 = Dropout(DROPOUT)(batch7)
    merge1 = Concatenate(axis=-1)([conv5, drop1])
    #
    up2 = conv2d_transpose_factory(conv_n_filters[3], kernel_size)((merge1))
    up2 = deconv_activation_layer(up2)
    batch8 = BatchNormalization(axis=-1)(up2)
    drop2 = Dropout(DROPOUT)(batch8)
    merge2 = Concatenate(axis=-1)([conv4, drop2])
    #
    up3 = conv2d_transpose_factory(conv_n_filters[2], kernel_size)((merge2))
    up3 = deconv_activation_layer(up3)
    batch9 = BatchNormalization(axis=-1)(up3)
    drop3 = Dropout(DROPOUT)(batch9)
    merge3 = Concatenate(axis=-1)([conv3, drop3])
    #
    up4 = conv2d_transpose_factory(conv_n_filters[1], kernel_size)((merge3))
    up4 = deconv_activation_layer(up4)
    batch10 = BatchNormalization(axis=-1)(up4)
    merge4 = Concatenate(axis=-1)([conv2, batch10])
    #
    up5 = conv2d_transpose_factory(conv_n_filters[0], kernel_size)((merge4))
    up5 = deconv_activation_layer(up5)
    batch11 = BatchNormalization(axis=-1)(up5)
    merge5 = Concatenate(axis=-1)([conv1, batch11])
    #
    up6 = conv2d_transpose_factory(1, kernel_size)((merge5))
    up6 = deconv_activation_layer(up6)
    batch12 = BatchNormalization(axis=-1)(up6)
    # Last layer to ensure initial shape reconstruction.

    up7 = Conv2D(
        1,
        (4, 4),
        dilation_rate=(2, 2),
        activation="sigmoid",
        padding="same",
        kernel_initializer=kernel_initializer,
    )((batch12))
    return up7


def separate_from_audio(audio,rate,mask_model,wiener_filter=True, return_spectrogram = False):
    split_stft, full_stft = preprocess_audio_tf(np.expand_dims(audio,axis=0),test=True)
    mask = mask_model.predict(split_stft)
    #objective = preprocess(np.array(np.hstack(objective_vocal_samples)),X_mean,X_std)
    mask_in_shape = np.concatenate(mask,axis=1)[:,:,0]
    input_in_shape = full_stft
    json_path = '../norm_data_full.json'
    with open(json_path) as infile:
        norm_data = json.load(infile)
    X_mean = norm_data['X_min']
    X_std = norm_data['X_max'] - norm_data['X_min']
    test_sample = np.zeros((513,input_in_shape.shape[1]), dtype=complex)
    test_sample[0:513] = full_stft[0:513]
    mask_final = np.zeros((513,test_sample.shape[1]))
    final_mag = np.zeros((513,test_sample.shape[1]))

    mask_final[0:512] = np.concatenate(mask,axis=1)[:,:,0]
    pre_result= preprocess(test_sample,X_mean,X_std)


    final_mag = denormalize(mask_final,X_mean,X_std)
    result_stft = np.multiply(np.exp(final_mag),np.exp(1j*np.angle(test_sample)))

    audio_vocal_pred = tf.signal.inverse_stft(result_stft.T,frame_length=1024, 
    frame_step=512,
    fft_length=1024,
    window_fn=tf.signal.inverse_stft_window_fn(512)).numpy()
    if wiener_filter:
        test_sample_T = test_sample.T[:,:,np.newaxis]
        result_stft_T = result_stft.T[:,:,np.newaxis,np.newaxis]


        v = norbert.contrib.residual_model(np.abs(result_stft_T),test_sample_T)
        result_wiener = norbert.wiener(v,test_sample_T,iterations=2)[:,:,:,0]
        result_stft = result_wiener.T.reshape(final_mag.shape[0],final_mag.shape[1])
        audio_vocal_pred = tf.signal.inverse_stft(result_stft.T,frame_length=1024, 
        frame_step=512,
        fft_length=1024,
        window_fn=tf.signal.inverse_stft_window_fn(512)).numpy()

    
    if return_spectrogram:
        return result_stft,mask_in_shape,preprocess_tf(input_in_shape,X_mean,X_std)
    return audio_vocal_pred[:len(audio)]



def load_unet_spleeter(kernel_size,weights_path):
    with tf.device('GPU'):
        init = keras.initializers.glorot_normal(seed=None)
        reg = 5e-5
        regularizer = tf.keras.regularizers.l2(reg)
        freq_bins = 512
        sample_len = 128
        l_input = Input(shape=(512, 128, 1))
        if kernel_size != (5,5):
            l_out_1 = get_unet_spleeter(l_input, kernel_size=kernel_size)
            l_out_2 = get_unet_spleeter(l_input, kernel_size=(kernel_size[1], kernel_size[0]))
            concat_layer = concatenate([l_out_1, l_out_2])
            mask_layer = Conv2D(1, (1, 1))(concat_layer)
            final_layer = Multiply()([l_input, mask_layer])
        else:
            mask_layer = get_unet_spleeter(l_input, kernel_size=(5, 5))
            final_layer = Multiply()([l_input, mask_layer])
        model = Model(inputs=[l_input], outputs=[final_layer])
        model.load_weights(weights_path)
    return model


def separate_umx(
    audio,
    rate=None,
    model_str_or_path="umxhq",
    targets=None,
    niter=1,
    residual=False,
    wiener_win_len=300,
    aggregate_dict=None,
    separator=None,
    device=None,
    filterbank="torch",
):
    """
    Open Unmix functional interface
    Separates a torch.Tensor or the content of an audio file.
    If a separator is provided, use it for inference. If not, create one
    and use it afterwards.
    Args:
        audio: audio to process
            torch Tensor: shape (channels, length), and
            `rate` must also be provided.
        rate: int or None: only used if audio is a Tensor. Otherwise,
            inferred from the file.
        model_str_or_path: the pretrained model to use
        targets (str): select the targets for the source to be separated.
            a list including: ['vocals', 'drums', 'bass', 'other'].
            If you don't pick them all, you probably want to
            activate the `residual=True` option.
            Defaults to all available targets per model.
        niter (int): the number of post-processingiterations, defaults to 1
        residual (bool): if True, a "garbage" target is created
        wiener_win_len (int): the number of frames to use when batching
            the post-processing step
        aggregate_dict (str): if provided, must be a string containing a '
            'valid expression for a dictionary, with keys as output '
            'target names, and values a list of targets that are used to '
            'build it. For instance: \'{\"vocals\":[\"vocals\"], '
            '\"accompaniment\":[\"drums\",\"bass\",\"other\"]}\'
        separator: if provided, the model.Separator object that will be used
             to perform separation
        device (str): selects device to be used for inference
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """
    if separator is None:
        separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            targets=targets,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True,
            filterbank=filterbank,
        )
        separator.freeze()
        if device:
            separator.to(device)

    if rate is None:
        raise Exception("rate` must be provided.")

    if device:
        audio = audio.to(device)
    audio = utils.preprocess(audio, rate, separator.sample_rate)

    # getting the separated signals
    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)
    return estimates



