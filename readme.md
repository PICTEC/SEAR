# SEAR - speech expansion and reconstruction

This repository contains code used in SEAR project, which ended in [ICASSP 2019 submission][https://www.cmsworkshops.com/ICASSP2019/Papers/Uploads/Proposals/PaperNum/4956/20181030064602_672782_4956.pdf].

### Code structure

Code related to the paper itself is contained in form of Jupyter notebooks in `notebooks` catalogue. The crucial ones that show training related to models presented in the paper are prefixed with `HYPO`. We've removed superfluous notebooks, but they're hidden in the git history.
Note that results in the notebooks may be overriden by more recent runs. We tried to keep the code intact.

The project has developed it's own legacy (sic!) code that is contained withing `loaders` folder. In the same 

For the sake of evaluation, we've contained several recordings and models in `evaluation` folder. Jupyter notebooks for using these models and showing results are stored in this folder. Binary data associated with this folder is in 

### Installation and usage

All necessary packages should be in `requirements.txt` file. For tensorflow, please supply it yourself, as I've installed it from source.

Majority of the notebooks use "Python3.6" kernel. This is due to a system upgrade that bumped up Python version during research. If your IPython doesn't have a kernel for 3.6, make one using e.g. [this tutorial][https://stackoverflow.com/questions/28831854/how-do-i-add-python3-kernel-to-jupyter-ipython]

### Sources:

We've used LibriSpeech database. For noise, we scraped our own database from the Internet. Not every recordings from the database was used.

##### Noise sources and attributions:

A mix of Attribution and Zero CC licences.

```
5 Minutes of Poznań by Jamafel - https://freesound.org/people/Jamafel/packs/20110/
8mm Projector by NemoDaedalus - https://freesound.org/people/nemoDaedalus/sounds/77621/
13G klauzury sports pack by 13GPanska_Lakota_Jan - https://freesound.org/people/13GPanska_Lakota_Jan/packs/21267/
140610thunderstorm by csengeri - https://freesound.org/people/csengeri/sounds/115521/
14F Panska Kaiprova_Johana by 14FPanskaKaiprova_Johana - https://freesound.org/people/14FPanskaKaiprova_Johana/sounds/419948/
1840 Grandfather Clock Retake by daveincamas - https://freesound.org/people/daveincamas/packs/2785/
1-minute Calcutta by Calcuttan - https://freesound.org/people/Calcuttan/packs/11448/
street scenes by saphe - https://freesound.org/people/saphe/packs/10916/
white noise woods NL by klankbeeld - https://freesound.org/people/klankbeeld/packs/7492/
soccer in Holland by klankbeeld - https://freesound.org/people/klankbeeld/packs/7946/
wind chimes by minian89 - https://freesound.org/people/minian89/sounds/217800/
room noise by apolloaiello - https://freesound.org/people/apolloaiello/sounds/329046/
industrial noise by klankbeeld https://freesound.org/people/klankbeeld/sounds/213228/
Crowd Noises by lonemonk - https://freesound.org/people/lonemonk/packs/1948/
marina by klanbeeld - https://freesound.org/people/klankbeeld/sounds/170563/
The Office by qubodup - https://freesound.org/people/qubodup/sounds/211945/
My Dog George by ronfront - https://freesound.org/people/ronfont/sounds/30344/
Dogs by felix.blume - https://freesound.org/people/felix.blume/packs/8598/
Car start and drive by han1 - https://freesound.org/people/han1/sounds/19025/
Field Recordings - Driving a Car by RutgerMuller - https://freesound.org/people/RutgerMuller/packs/3270/
The Sea by digifishmusic - https://freesound.org/people/digifishmusic/packs/2617/
FrogsAndCrickets_ExcerptB_JMA_24Bit_48k.wav by Greysound - https://freesound.org/people/greysound/sounds/32655/
```

### Final words

We plan to utilize some of the discoveries associated with the paper in development of an upgraded model. 

Contact information: `pawel.tomasik@pictec.eu`
