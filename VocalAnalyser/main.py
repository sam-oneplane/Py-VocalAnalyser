from examples.librosa_examples import librosa_sample
from analyser.Seperator import vocal_seperator
from analyser.FormatConverter import format_convertor
import sys
import os

if __name__ == '__main__':
    #librosa_sample()
    #mp3_file = sys.argv[1]
    audio_path = sys.argv[1]
    audio_name  = sys.argv[2]
    #wav_file = os.path.join(audio_path, os.path.splitext(os.path.basename(mp3_file))[0] + ".wav")
    #convertor = format_convertor.AudioWavConvertor(wav_file, "wav")
    #convertor.apply(mp3_file)
    seperator = vocal_seperator.AudioSeperator(audio_path, audio_name, is_mono=True)
    seperator.spectral_subtraction()
    seperator.noise_reduction('wav_samples/out_vocal.wav', 'wav_samples/vocal_noisless.wav')
    
