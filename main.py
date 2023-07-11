import os
import glob
import sounddevice as sd
import speaker_verification_toolkit.tools as svt
from scipy.io.wavfile import write
import soundfile as sf
import math
import librosa
import pyworld
import pysptk
import numpy as np

corpus_path = "./Pashto_Speech_Corpus/"
file_list = os.listdir(corpus_path)
exclude = ["gloss.txt", "tmp", "user", ".DS_Store", "ref"]
SAMPLING_RATE = 44100
FRAME_PERIOD = 5.0
alpha = 0.65  # commonly used at 22050 Hz
fft_size = 512
mcep_size = 34

def id_linker():
    id2word = {}
    word2wavs = {}
    ids = []

    for file in file_list:
        if file not in exclude:
            split = file.split("-")
            word_id = int(split[0])
            ids.append(word_id)
            id2word[word_id] = split[1]
            path = glob.glob(corpus_path + file + "/*")
            word2wavs[word_id] = path
    ids.sort()
    return ids, id2word, word2wavs


# Load a wav file with librosa.
def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav


def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y

    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


def wav2mcep_numpy(wavfile, target_directory, alpha=0.65, fft_size=512, mcep_size=34):
    # make relevant directories
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    loaded_wav = load_wav(wavfile, sr=SAMPLING_RATE)

    # Use WORLD vocoder to spectral envelope
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=SAMPLING_RATE, frame_period=FRAME_PERIOD,
                                 fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    fname = os.path.basename(wavfile).split("/")[-1][:-4]
    np.save(os.path.join(target_directory, fname + '.npy'), mgc, allow_pickle=False)


# obtain all the mceps and create the directories hosting them
def make_mceps(word2wavs):
    for word in word2wavs:
        new_dir = corpus_path + "tmp/" + str(word)
        # don't do the whole shebang if we suspect it exists already
        if not os.path.isdir(new_dir):
            for wav in word2wavs[word]:
                if wav.split(".")[-1] == "wav" and wav.split("/")[-1].split("_")[0][-1] == "Y":
                    wav2mcep_numpy(wav, new_dir, fft_size=fft_size, mcep_size=mcep_size)
    silence = corpus_path + "ref/silence.wav"
    silence_dir = corpus_path + "ref/mcep"
    if not os.path.isdir(silence_dir):
        wav2mcep_numpy(silence, silence_dir, fft_size=fft_size, mcep_size=mcep_size)



def average_mcd(ref_mcep_files, synth_mcep_files, cost_function):
    """
    Calculate the average MCD.
    :param ref_mcep_files: list of strings, paths to MCEP target reference files
    :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
    :param cost_function: distance metric used
    :returns: average MCD, total frames processed
    """
    min_cost_tot = 0.0
    frames_tot = 0

    for ref in ref_mcep_files:
        for synth in synth_mcep_files:
            # get the trg_ref and conv_synth speaker name and sample id
            ref_fsplit, synth_fsplit = os.path.basename(ref).split('_'), os.path.basename(synth).split('_')
            ref_spk, ref_id = ref_fsplit[0], ref_fsplit[-1]
            synth_spk, synth_id = synth_fsplit[0], synth_fsplit[-1]

            # if the speaker name is the same and sample id is the same, do MCD

            # load MCEP vectors
            ref_vec = np.load(ref)
            ref_frame_no = len(ref_vec)
            synth_vec = np.load(synth)

            # dynamic time warping using librosa
            min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T,
                                               metric=cost_function)

            min_cost_tot += np.mean(min_cost)
            frames_tot += ref_frame_no

    mean_mcd = min_cost_tot / frames_tot

    return mean_mcd, frames_tot


# create a valid range of MCD vals for every word
def make_mcep_range():
    range = {}
    for word in word2wavs:
        mceps = glob.glob(corpus_path + "tmp/" + str(word) + "/*")
        mcep1 = [mceps[0]]
        mcep2 = [mceps[1]]
        sil = ['./Pashto_Speech_Corpus/ref/mcep/silence.npy']
        mean_mcd, tot = average_mcd(mcep1, mcep2, log_spec_dB_dist)
        sil_mcd, tot = average_mcd(mcep1, sil, log_spec_dB_dist)
        range[word] = [sil_mcd, mean_mcd]
    return range

print("Loading...")
# get 1) a sorted list of the number of words 2) a dict mapping id:word 3) a dict mapping word: wav paths
ids, id2word, word2wavs = id_linker()
# obtain the mceps from these words and store them to the tmp directory
make_mceps(word2wavs)
mcep_range = make_mcep_range()


# set up parameters for recording
user_sr = 44100
user_duration = 2
cost_function = log_spec_dB_dist

print("Loaded.")

# enter loop
while True:

    # greet user, introduce list of possible practice words
    print("Which word would you like to practice? \n")
    for word_id in ids:
        print(str(word_id) + "-" + str(id2word[word_id]))
    print("\n")


    # get user preferred word
    word_id = int(input("Please input the number prefix: "))
    word = id2word[word_id]

    prac_choice = int(input("Press 0 to hear a native pronunciation of this word. Press 1 to practice it. Press 2 to "
                            "return to the practice word list: "))

    while prac_choice != 2:

        if prac_choice == 0:
            native_wav = word2wavs[word_id][0]
            native, fs = sf.read(native_wav)
            sd.play(native, fs)
            sd.wait()
            print("Press 0 to hear a native pronunciation of this word. Press 1 to practice it. "
                  "Press 2 to return to the practice word list:")

        elif prac_choice == 1:
            # get user speech input, convert the NumPy array to an audio file with the given sampling frequency
            recording = sd.rec(int(user_duration * user_sr), samplerate=user_sr, channels=2)
            print("Listening...")
            sd.wait()
            print("Done listening. \n Scoring...")
            user_path = corpus_path + "user"
            write(user_path + "/recording.wav", user_sr, recording)

            # remove silence ?
            recording, sampling_rate = librosa.load(user_path + "/recording.wav", sr=44100, mono=True)
            recording = svt.rms_silence_filter(recording)
            sf.write(user_path + "/recording.wav", recording, 44100)

            # get user production mceps
            wav2mcep_numpy(user_path + "/recording.wav", user_path + "/user_mcep", fft_size=fft_size, mcep_size=mcep_size)

            references = glob.glob(corpus_path + "tmp/" + str(word_id) + "/*")
            user_mceps = glob.glob(user_path + "/user_mcep/*")

            # calc mcd & scale it
            mcd, tot_frames_used = average_mcd(references, user_mceps, cost_function)
            top, bottom = mcep_range[word_id][0], mcep_range[word_id][1]

            if mcd > top:
                print("You scored 0%. Try practicing again.")
            if mcd < top and mcd > bottom:
                score = int(100 - ((mcd - bottom / top - bottom) * 100))
                print("You scored " + str(score) + "%! You can continue practicing by pressing 1, return to the word "
                                                   "list by pressing 2, or hear a native production by pressing 0: ")

            if mcd < bottom:
                print("You scored 100%! You can continue practicing by pressing 1, return to the word "
                                                   "list by pressing 2, or hear a native production by pressing 0: ")

            # delete the recording
            os.remove(user_path + "/recording.wav")

        else:
            print("That's not a valid choice. Press 0 to hear a native pronunciation of this word. "
                  "Press 1 to practice it. Press 2 to return to the practice word list.")

        prac_choice = int(input())