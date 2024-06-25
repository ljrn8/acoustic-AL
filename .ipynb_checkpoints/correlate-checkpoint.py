from util import *

def cross_corr_deployment(
        reference_file, deployment=1, site=1, sr=None, n_mfcc=13, 
        threshold=0.99, output_file="./correlations/test", max_corrs=100_000
    ):
    """ Cross correlate MFCC's of the reference audio segment against all recordings in the given deployment.
    outputs correlations as (file, time) under ./correlations/test. (Needs to fixed)
    """

    data_dir = path.join(get_depl_dir(deployment, site), "Data")

    # load and process reference segment
    ref_y, ref_sr = load(reference_file, sr=sr)
    ref_mfcc_flat = librosa.feature.mfcc(y=ref_y, sr=ref_sr, n_mfcc=n_mfcc).flatten()
    reference_duration = librosa.get_duration(y=ref_y, sr=ref_sr)

    all_matches = []

    # search directory for wavs
    for root, _, files in os.walk(data_dir):
        for i, file in enumerate(files):
            if file.endswith(".wav"):
                print(f"correlating {file} | {i} / {len(files)}")
                file_path = os.path.join(root, file)
                
                # large audio segment
                y, sr = load(file_path, sr=sr)

                # get flat MFCCs of large audio segment (10min)
                mfcc_flat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()

                # corrolate MFCCs
                correlation = np.correlate(mfcc_flat, ref_mfcc_flat, mode='valid') # TODO valid or full?
                
                # get high corrolation matches
                matches_indexes = np.where(correlation > threshold * np.max(correlation))[0]

                # get matches in exact time
                for match_index in matches_indexes:
                    match_time = librosa.frames_to_time(match_index, sr=ref_sr)
                    print(match_time, end = " ")
                    all_matches.append((file_path, match_time))
                    
                    # stop if max correlations reached
                    if len(all_matches) >= max_corrs:
                        return all_matches
                    
                    # write
                    with open(output_file, "a") as f:
                        f.write(f"{match_time}, {file}\n")
                
                print("\ncurrent matches", len(all_matches)) # wrong (x2)

    return all_matches