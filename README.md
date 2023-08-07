# Keystroke-Datasets
A public repo for the datasets recorded as part of "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards".

## Zoom dataset
- Recorded via the built-in `record meeting` function in the web-conferencing app Zoom.

## Phone-Recorded dataset
- Recorded by an iPhone 13 mini set down next to a 16-inch M1 Pro MacBook Pro, on top of a microfibre cloth to reduce vibration

## File structure
- Each dataset consists of `.wav` files for the keys 1-0, q-m on the keyboard, each file representing 25 keystrokes at slightly differing intervals and strengths.
- Each keystroke was made in the same environment by the same person

## Isolating Keystrokes
- Please look at the below function for isolating keystrokes:
```Python3
def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    strokes = []
    # -- signal'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(signal, sr=sample_rate)
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    # norm = np.linalg.norm(energy)
    # energy = energy/norm
    # -- energy'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(energy)
    threshed = energy > threshold
    # -- peaks'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(threshed.astype(float))
    peaks = np.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate*0.1*(-1)
    # '-- isolating keystrokes'
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*scan) + size//2
        if timestamp > prev_end + (0.1*sample_rate):
            keystroke = signal[timestamp-before:timestamp+after]
            strokes.append(torch.tensor(keystroke)[None, :])
            if show:
                plt.figure(figsize=(7, 2))
                librosa.display.waveshow(keystroke, sr=sample_rate)
            prev_end = timestamp+after
    return strokes
```
- I used this function as follows to create a dataframe of the Zoom-recorded keystrokes:
```Python3
AUDIO_FILE = './Zoom/'
keys_s = '1234567890qwertyuiopasdfghjklzxcvbnm'
labels = list(keys_s)
keys = [k + '.wav' for k in labels]
data_dict = {'Key':[], 'File':[]}

for i, File in enumerate(keys):
    loc = AUDIO_FILE + File
    samples, sample_rate = librosa.load(loc, sr=None)
    #samples = samples[round(1*sample_rate):]
    strokes = []
    prom = 0.06
    step = 0.005
    while not len(strokes) == 25:
        strokes = isolator(samples[1*sample_rate:], sample_rate, 48, 24, 2400, 12000, prom, False)
        if len(strokes) < 25:
            prom -= step
        if len(strokes) > 25:
            prom += step
        if prom <= 0:
            print('-- not possible for: ',File)
            break
        step = step*0.99
    label = [labels[i]]*len(strokes)
    data_dict['Key'] += label
    data_dict['File'] += strokes

df = pd.DataFrame(data_dict)
mapper = {}
counter = 0
for l in df['Key']:
    if not l in mapper:
        mapper[l] = counter
        counter += 1
df.replace({'Key': mapper}, inplace=True)
```
