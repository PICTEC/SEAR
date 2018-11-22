import nltk
import os
import subprocess

vocab = nltk.corpus.cmudict.dict()

def convert(src, target):
    subprocess.Popen(['sox', src, '-b', '16', '-e', 'signed',
                      '-r', '16000', target]).communicate()

def sentence_to_phonemes(sent):
    phonemes = []
    for word in sent.lower().split():
        try:
            phonemes += vocab[word][0]
        except KeyError:
            phonemes += "OOV"
    phonemes = ["".join([ch for ch in x if ch.isalpha()]) for x in phonemes]
    return phonemes

def find_in_transcript(transcript, fname):
    sentname = fname[:-5]
    with open(transcript) as f:
        data = f.read().split('\n')
    tr = [line for line in data if line.startswith(sentname)]
    print(tr[0])
    return " ".join(tr[0].split(' ')[1:])

def load_librispeech(path, targetpath):
    index = 0
    try:
        os.mkdir(targetpath)
    except:
        pass
    for session in os.listdir(path):
        for part in os.listdir(os.path.join(path, session)):
            transcriptname = os.path.join(path, session, part, 
                "{}-{}.trans.txt".format(session, part))
            for recording in os.listdir(os.path.join(path, session, part)):
                if recording.endswith(".flac"):
                    text = find_in_transcript(transcriptname, recording)
                    text = sentence_to_phonemes(text)
                    convert(os.path.join(path, session, part, recording),
                        os.path.join(targetpath, "{}.wav".format(index)))
                    with open(os.path.join(targetpath, "{}.txt".format(index)), "w") as f:
                        f.write(" ".join(text))
                    index += 1
    return text

if __name__ == "__main__":
    load_librispeech("/home/zantyr/Downloads/LibriSpeech/train-clean-100",
        "DAE-libri")
