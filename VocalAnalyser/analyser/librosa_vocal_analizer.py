import librosa
import numpy as np
import matplotlib.pyplot as plt

class Vocal:
    def __init__(self, audio_path:str):
        # Load the reference vocal .wav file
        stereo_audio, sr = librosa.load(audio_path, mono=False)
        print(f'Sample Ratio: {sr},  Channels: {len(stereo_audio)}')
        self._sr = sr
        self._left_channel: np.ndarray = stereo_audio[0]
        self._right_channel: np.ndarray = stereo_audio[1]

    @property
    def sr(self):
        return self._sr

    @property
    def channels(self):
        return (self._left_channel, self._right_channel)



class Analyser:

    def __init__(self, sr: float, left_ch: np.ndarray, right_ch: np.ndarray):
        self._sr = sr
        self.left_channel: np.ndarray = left_ch
        self.right_channel: np.ndarray = right_ch

    # transform to freq domain using DFT (power spectrum over freq domain)
    @classmethod
    def stft_analyzer(cls, vocal: Vocal):
        n_fft = 512
        left_ch, right_ch = vocal.channels
        return (np.abs(librosa.stft(left_ch, n_fft)),
                np.abs(librosa.stft(right_ch, n_fft)))

    # time-freq spectrogram (iDFT on log power spectrum)
    @classmethod
    def amp2db_analyzer(cls, left_channel: np.ndarray, right_channel: np.ndarray):
        return (
            librosa.amplitude_to_db(abs(left_channel)),
            librosa.amplitude_to_db(abs(right_channel))
        )

    @classmethod
    def mfcc_analyzer(cls, vocal: Vocal):
        left_channel, right_channel = vocal.channels
        return ( librosa.feature.mfcc(y=left_channel, sr=vocal.sr, n_mfcc=13),
                 librosa.feature.mfcc(y=right_channel, sr=vocal.sr, n_mfcc=13))


    @classmethod
    def pitch_analyzer(cls, vocal: Vocal):
        def pitch_values(pitches: np.ndarray, magnitudes: np.ndarray,  values: list):
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    values.append(pitch)


        left_channel, right_channel = vocal.channels
        pitches_left, mag_left = librosa.core.piptrack(y=left_channel, sr=vocal.sr)
        pitches_right, mag_right = librosa.core.piptrack(y=right_channel, sr=vocal.sr)

        pitch_left_values = []
        pitch_right_values = []
        pitch_values(pitches_left, magnitudes=mag_left, values=pitch_left_values)
        pitch_values(pitches_right, magnitudes=mag_right, values=pitch_right_values)
        return (pitch_left_values, pitch_right_values)


## PLOT
def librosa_wave_plot(data: np.ndarray, sampling_rate: float, title: str, labels: tuple):
    plt.figure(figsize=(12,4))
    librosa.display.waveshow(data, sr=sampling_rate)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def librosa_simple_plot(data: np.ndarray, title: str, labels: tuple):
    plt.figure(figsize=(12,4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def librosa_spec_plot(data: np.ndarray, sampling_rate: float, title: str, labels: tuple):
    plt.figure(figsize=(12,4))
    librosa.display.specshow(data, sr=sampling_rate, hop_length=512, x_axis=labels[0], y_axis=labels[1])
    plt.title(title)
    plt.colorbar(format='%+2.0f db')
    plt.show()

def librosa_mfcc_plot(data: np.ndarray, sampling_rate: float, title: str, label: str):
    plt.figure(figsize=(12,4))
    librosa.display.specshow(data, sr=sampling_rate, x_axis=label)
    plt.title(title)
    plt.colorbar(format='%+2.0f db')
    plt.show()




import os
if __name__ == '__main__':

    path = os.path.join("./Samples", "Zeno_Signs_vocals.wav")
    vocal : Vocal = Vocal(path)
    left_channel, right_channel = vocal.channels
    xyLabels = ('Time', 'Amplitude')
    librosa_wave_plot(left_channel, vocal.sr, 'Left Channel Wave', xyLabels)
    librosa_wave_plot(right_channel, vocal.sr, 'Right Channel Wave', xyLabels)

    ## freq domain analysis (DFT)
    left_ft, right_ft = Analyser.stft_analyzer(vocal)
    xyLabels = ('Freq', 'Amplitude')
    librosa_simple_plot(left_ft, 'Left Channel Spectrum', xyLabels)
    librosa_simple_plot(right_ft, 'Right Channel Spectrum', xyLabels)

    ## Time Frequency represention
    left_db, right_db = Analyser.amp2db_analyzer(left_ft, right_ft)
    librosa_spec_plot(left_db, vocal.sr, 'Left Channel Spectogram', ('time', 'linear'))
    librosa_spec_plot(right_db, vocal.sr, 'Right Channel Spectogram', ('time', 'linear'))

    ## Mel Frequency Ceptral Coeff (MFCC)
    left_mfcc, right_mfcc = Analyser.mfcc_analyzer(vocal)
    librosa_mfcc_plot(left_mfcc, vocal.sr, 'Left Channel MFCC', 'time')
    librosa_mfcc_plot(right_mfcc, vocal.sr, 'Right Channel MFCC', 'time')

    ## Pitch detection
    left_pitch, right_pitch = Analyser.pitch_analyzer(vocal)
    xyLabels = ('Time', 'Pitch(Hz)')
    librosa_simple_plot(np.array(left_pitch), 'Left Chnnel Pitch Detection', xyLabels)
    librosa_simple_plot(np.array(right_pitch), 'Right Channel Pitch Detection', xyLabels)
