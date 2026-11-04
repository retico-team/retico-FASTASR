# retico-FASTASR
## Install Requirements

``` bash
git clone https://github.com/allenai/OLMoASR.git
pip install -r requirements/requirements.txt
pip install -e .

```
## Runner Example

``` python
import sys, os
from retico import *
from retico_core import *

os.environ['RESPEAKER'] = 'retico-respeakermic'
os.environ['RETICO'] = 'retico-core'
os.environ['FASTASR'] = 'retico-FASTASR'

sys.path.append(os.environ['RETICO'])
sys.path.append(os.environ['FASTASR'])

from retico_core.debug import CallbackModule
from retico_fastasr.fastasr import FastASRModule
from retico_core.audio import MicrophoneModule
msg = []

def callback(update_msg):
    global msg
    for x, ut in update_msg:
        if ut == UpdateType.ADD:
            msg.append(x)
        if ut == UpdateType.REVOKE:
            msg.remove(x)
    txt = ""
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed
    print(" " * 80, end="\r")
    print(f"{txt}", end="\r")
    if committed:
        msg = []
        print("") 

mic = MicrophoneModule(rate=16000)
asr = FastASRModule(framerate=16000, silence_dur=1, language="en", task="transcribe")
cb = CallbackModule(callback)

mic.subscribe(asr)
asr.subscribe(cb)

mic.run()
asr.run()
cb.run()

print("Running...")
input()

mic.stop()
asr.stop()
cb.stop()

```

## Citations

```
@misc{ngo2025olmoasropenmodelsdata,
      title={OLMoASR: Open Models and Data for Training Robust Speech Recognition Models}, 
      author={Huong Ngo and Matt Deitke and Martijn Bartelds and Sarah Pratt and Josh Gardner and Matt Jordan and Ludwig Schmidt},
      year={2025},
      eprint={2508.20869},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.20869}, 
}
```
