import heapq
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from plotting import plot_correlations, view_spectogram
from scipy.signal import correlate2d
from util import *


#### !!
def template_match(template="./data/templates/up_down_down.txt"):
    
    from maad import sound, util
    from maad.rois import template_matching

    with open(template) as f:
        start_s, end_s, name, f_low, f_high = f.readline().split()

    # Set spectrogram parameters
    tlims = (start_s, end_s)
    flims = (f_low, f_high)
    nperseg = 1024
    noverlap = 512
    window = 'hann'
    db_range = 80

    # load data
    large_audio_file = path.join(get_depl_dir(1,1), "Data", "1_20230316_063000.wav")
    s, fs = sound.load(large_audio_file)

    # Compute spectrogram for template signal
    Sxx_template, _, _, _ = sound.spectrogram(s, fs, window, nperseg, noverlap, flims, tlims)
    Sxx_template = util.power2dB(Sxx_template, db_range)

    # Compute spectrogram for target audio
    Sxx_audio, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap, flims)
    Sxx_audio = util.power2dB(Sxx_audio, db_range)
    
    peak_th = 0.3 # set the threshold to find peaks
    xcorrcoef, rois = template_matching(Sxx_audio, Sxx_template, tn, ext, peak_th)
    rois['min_f'] = flims[0]
    rois['max_f'] = flims[1]
    print(rois)
    
    """ also consider 
    # TODO restrict frequency
    
    # Load audio file
    audio_data, sr = librosa.load('forest_audio.wav', sr=44100)

    # Extract features (e.g., Mel spectrogram)
    mel_spec = librosa.feature.melspectrogram(audio_data, sr=sr)

    # Load template (example birdcall template)
    template_data, _ = librosa.load('birdcall_template.wav', sr=sr)
    template_spec = librosa.feature.melspectrogram(template_data, sr=sr)

    # Perform template matching
    result = cv2.matchTemplate(mel_spec, template_spec, cv2.TM_CCOEFF_NORMED)

"""

template_match()




def _compute_mfcc_from_restricted_stft(
    S_restricted,
    sr,
    frequency_range,
    n_mfcc=13,
    n_mels=128,
):
    fmin, fmax = frequency_range

    # Define the Mel filter bank to match the restricted frequency range
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=S_restricted.shape[0] * 2 - 2, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    S_mel = np.dot(mel_basis, np.abs(S_restricted) ** 2)

    # Compute the log Mel spectrogram
    log_S_mel = librosa.power_to_db(S_mel)

    # Compute MFCCs from the log Mel spectrogram
    mfcc = librosa.feature.mfcc(S=log_S_mel, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def _correlate(S, S_ref):

    # normalize first
    S = (S - np.mean(S)) / np.std(S)
    S_ref = (S_ref - np.mean(S_ref)) / np.std(S_ref)

    # print(S.shape, S_ref.shape)
    return correlate2d(S, S_ref)  # valid


def cross_correlate(
    reference_file,
    frequency_range,
    deployment=1,
    site=1,
    sr=None,
    threshold=0.99,
    output_file=path.join(
        "correlations", "corr_" + datetime.now().strftime("%m.%d.%Y.%H.%M.%S")
    ),
    return_correlations=15,
    max_recordings=3,
):
    print("outputting to ", output_file)
    Path(output_file).touch()
    data_dir = path.join(get_depl_dir(deployment, site), "Data")

    # load and process reference segment
    ref_y, ref_sr = load(reference_file, sr=sr)
    reference_duration = librosa.get_duration(y=ref_y, sr=ref_sr)

    # constrict frequency range
    S_ref = freq_range(librosa.stft(ref_y), ref_sr, frequency_range)

    all_correlations = []

    # search directory for wavs
    for root, _, files in os.walk(data_dir):
        for i, file in enumerate(files):
            if file.endswith(".wav"):
                print(f"preparing {file} | {i} / {len(files)}", end="\r")
                file_path = os.path.join(root, file)

                # large audio segment
                y, sr = load(file_path, sr=sr)
                S = freq_range(librosa.stft(y), sr, frequency_range)

                S_mfcc_flat = _compute_mfcc_from_restricted_stft(
                    S, sr, frequency_range, n_mels=64
                ).flatten()
                S_ref_mfcc_flat = _compute_mfcc_from_restricted_stft(
                    S_ref, sr, frequency_range, n_mels=64
                ).flatten()

                # correlate flat Spectrograms (?)
                print(f"correlating {file} | {i} / {len(files)}", end="\r")
                correlation = np.correlate(S_mfcc_flat, S_ref_mfcc_flat)  # 1D corr

                # NOTE per file
                # matches_indexes = np.where(correlation > threshold * np.max(correlation))[0]
                all_correlations.append({"corr": correlation, "file": file})

                if max_recordings and i > max_recordings:
                    print("\n")
                    break

    # find the top n correlations with a max heap
    heap = [(0, 0, None)] * return_correlations
    heapq.heapify(heap)
    for i, item in enumerate(all_correlations):
        for ii, x in enumerate(item["corr"]):
            if not heap[0] or x > heap[0][0]:
                match_time = librosa.frames_to_time(ii, sr=ref_sr)
                heapq.heapreplace(heap, (x, match_time, item["file"]))

    with open(output_file, "a") as f:
        for item in heap:
            f.write(",".join([str(i) for i in item]) + "\n")

    return heap


def splice_correlations(
    correlations_file,
    reference_wav=None,
    duration=None,
    output_dir="./correlations/segments/",
    deployment=1,
    site=1,
):
    """produces the correleations (to be annotated) as wav files for verification
    **ideally should need to use this
    """
    if not (reference_wav or duration):
        return ValueError("require wav or duration in seconds")

    # get the segment length
    if reference_wav:
        print("inferring reference length from given file")
        y, sr = librosa.load(reference_wav)
        duration = librosa.get_duration(y=y, sr=sr)

    # read correlations
    with open(correlations_file) as f:
        for line in f:
            time, file = line.split(", ")
            time = float(time)
            file = file.strip()

            # find the correlation segment in the original file
            recording_path = get_depl_dir(deployment, site) + "/Data/" + file
            extract_segment(
                recording_path,
                output_dir + "corr_" + file[:3] + "_" + str(time) + ".wav",
                time,
                time + duration,
            )


def correlate_save_view(
    reference_data,
    reference_file,
    output_file,
    return_correlations=30,
    max_recordings=50,
):

    # read reference data
    with open(reference_data, "r") as f:
        file, start_s, end_s, name, start_hz, end_hz = f.readline().split()

    start_s, end_s, start_hz, end_hz = [
        float(i) for i in [start_s, end_s, start_hz, end_hz]
    ]
    time_segment, frequency_range = (start_s, end_s), (start_hz, end_hz)
    print("reference data: ", file, name, time_segment, frequency_range)
    file_path = path.join(path.join(get_depl_dir(1, 1), "Data"), file + ".wav")

    # view the spectrogram of the correlation reference
    print("\n== Reference ==")
    view_spectogram(
        file_path, time_segment=time_segment, frequency_range=frequency_range
    )

    # copy the reference segment locally
    if not os.path.exists(reference_file):
        extract_segment(file_path, reference_file, start_s, end_s)

    # correlate
    print("\n== Correlating ==")
    corrs = cross_correlate(
        reference_file,
        frequency_range,
        return_correlations=return_correlations,
        max_recordings=max_recordings,
        output_file=output_file,
    )

    # view correlations
    print("\n== Plotting ==")
    plot_correlations(
        output_file, reference_wav=reference_file, frequency_lines=frequency_range
    )

    return corrs
