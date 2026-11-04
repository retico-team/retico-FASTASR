"""
OlMo ASR Module
==================

This module provides on-device ASR capabilities by using the OlMo transformer
provided by huggingface. In addition, the ASR module provides end-of-utterance detection
(with a VAD), so that produced hypotheses are being "committed" once an utterance is
finished.
"""

import threading
import retico_core
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
import transformers
import pydub
import olmoasr
import webrtcvad
import numpy as np
import time

transformers.logging.set_verbosity_error()
from transformers import AutoModel

class FASTASR:
    def __init__(self, framerate=16_000, silence_dur=1, vad_agressiveness=3,
                 silence_threshold=0.75, model_size="base", **_ignored):
        self.model = olmoasr.load_model(model_size, inference=True)  # tiny -> OLMoASR-tiny.en
        self.audio_buffer = []
        self.framerate = framerate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold

    def _resample_audio(self, audio):
        if self.framerate != 16_000:
            # resample if framerate is not 16 kHz
            s = pydub.AudioSegment(
                audio, sample_width=2, channels=1, frame_rate=self.framerate
            )
            s = s.set_frame_rate(16_000)
            return s._data
        return audio
        
    def get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2
            self._n_sil_frames = int(self.silence_dur / (frame_length / 16_000))
        return self._n_sil_frames

    def recognize_silence(self):
        n_sil_frames = self.get_n_sil_frames()
        if not n_sil_frames or len(self.audio_buffer) < n_sil_frames:
            return True
        silence_counter = 0
        for a in self.audio_buffer[-n_sil_frames:]:
            if not self.vad.is_speech(a, 16_000):
                silence_counter += 1
        if silence_counter >= int(self.silence_threshold * n_sil_frames):
            return True
        return False
    
    def add_audio(self, audio):
        audio = self._resample_audio(audio)
        self.audio_buffer.append(audio)

    def recognize(self):
        silence = self.recognize_silence()

        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self.get_n_sil_frames():]

        if not self.vad_state:
            return None, False

        flags = [self.vad.is_speech(a, 16_000) for a in self.audio_buffer]
        speech_idx = [i for i, f in enumerate(flags) if f]

        if len(speech_idx) < 5: 
            if silence:
                self.vad_state = False
                self.audio_buffer = []
                return None, False
            return None, self.vad_state

        frame_s = len(self.audio_buffer[0]) / 2 / 16_000
        pad = int(0.3 / frame_s)
        start = max(0, speech_idx[0] - pad)
        end = min(len(self.audio_buffer), speech_idx[-1] + pad + 1)
        full_audio = b"".join(self.audio_buffer[start:end])

        npa = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
        if len(npa) < 10:
            return None, False

        result = self.model.transcribe(
            npa,
            fp16=False,                      # also silences the FP16 warning
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        segs = [s for s in result.get("segments", [])
                if s.get("no_speech_prob", 0) < 0.6 and s.get("avg_logprob", 0) > -1.0]
        transcription = " ".join(s["text"].strip() for s in segs)

        if silence:
            self.vad_state = False
            self.audio_buffer = []

        return transcription, self.vad_state
            

    def reset(self):
        self.vad_state = True
        self.audio_buffer = []    

class FastASRModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Fast ASR Module"
    
    @staticmethod
    def description():
        return "A module that recognizes speech using OlMo."
    
    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU
    
    def __init__(self, framerate=None, silence_dur=1, language=None, task="transcribe", **kwargs):
        super().__init__(**kwargs)

        self.acr = FASTASR(
            silence_dur=silence_dur,
            language=language,
            task=task,
        )
        self.framerate = framerate
        self.silence_dur = silence_dur
        self._asr_thread_active = False
        self.latest_input_iu = None

    def process_update(self, update_message):
        for iu, ut in update_message:
            # Audio IUs are only added and never updated.
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.framerate is None:
                self.framerate = iu.rate
                self.acr.framerate = self.framerate
            self.acr.add_audio(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _asr_thread(self):
        while self._asr_thread_active:
            time.sleep(0.5)
            if not self.framerate:
                continue
            prediction, vad = self.acr.recognize()
            if prediction is None:
                continue
            end_of_utterance = not vad
            um, new_tokens = retico_core.text.get_text_increment(self, prediction)

            if len(new_tokens) == 0 and vad:
                continue

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu(self.latest_input_iu)
                eou = i == len(new_tokens) - 1 and end_of_utterance
                output_iu.set_asr_results([prediction], token, 0.0, 0.99, eou)
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)

            if end_of_utterance:
                for iu in self.current_output:
                    self.commit(iu)
                    um.add_iu(iu, retico_core.UpdateType.COMMIT)
                self.current_output = []

            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        self._asr_thread_active = False
        self.acr.reset()