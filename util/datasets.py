# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import  json, random
import torch.utils.data as Data
import os
import torch
from lavin import Tokenizer
import copy
import pandas as pd
from util.misc import sample_frame_indices, read_video_pyav, load_video
from vivit import VivitImageProcessor
from whisper import WhisperFeatureExtractor
from transformers import AutoTokenizer, AutoProcessor
import av
import soundfile
import numpy as np
from pathlib import Path
import gc
from qwen_vl_utils import fetch_video

class MELDDataset(Data.Dataset):
    def __init__(self, args, path, split, emotion):
        super(MELDDataset, self).__init__()
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------        
        self.tokenizer = Tokenizer(model_path= args.llama_model_path + '8B/tokenizer.model')
        self.max_words = args.max_seq_len
        self.split=split
        self.emotion = emotion
        self.raw_data = pd.read_csv(path)
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        print(f"number of examples in split {split}: {len(self.raw_data)}\n")

    def tokenize(self,prompt,answer):
        example=prompt + ' ' + answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask
    
    def __getitem__(self, idx):
        data_point = self.raw_data.iloc[idx]
        video_path = data_point['video']
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        video = self.image_processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        audio_path = data_point['audio']
        waveform, sampling_rate = soundfile.read(audio_path)
        audio = self.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.squeeze(0)
        prompt_text = f"Text: {data_point['sentence']}."
        prompt_text+=" Response: "
        prompt_text='\n'+prompt_text
        prompt_text = prompt_text.replace("  ", " ").strip()
        prompt_answer = f"The {self.emotion} is {data_point[self.emotion]}."
        example, labels, example_mask, label_mask=self.tokenize(prompt_text,prompt_answer)
        gc.collect()
        return example, labels, example_mask, video, audio

    def __len__(self):
        return len(self.raw_data)

    def shuffle_list(self, list):
        random.shuffle(list)

class EmotionDatasetForRegression(Data.Dataset):
    def __init__(self, args, path, split, emotion, model_path, max_words=512):
        super(EmotionDatasetForRegression, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------        
        self.tokenizer = Tokenizer(model_path= model_path + '8B/tokenizer.model')
        self.max_words = max_words
        self.split=split
        self.emotion = emotion
        self.raw_data = pd.read_csv(path)
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        print(f"number of examples in split {split}: {len(self.raw_data)}\n")

    def tokenize(self,prompt,answer):
        example=prompt + ' ' + answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=False), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask
    
    def __getitem__(self, idx):
        data_point = self.raw_data.iloc[idx]
        video_path = data_point['video']
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=16, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        video = self.image_processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        audio_path = data_point['audio']
        waveform, sampling_rate = soundfile.read(audio_path)
        audio = self.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.squeeze(0)
        prompt_text = f"Text: {data_point['sentence']}"
        prompt_text+="Response: "
        prompt_text='\n'+prompt_text
        prompt_text = prompt_text.replace("  ", " ").strip()
        prompt_answer = f"The {self.emotion} is "
        example, labels, example_mask, label_mask=self.tokenize(prompt_text,prompt_answer)
        gc.collect()
        return example, labels, torch.tensor(data_point['y']), example_mask, video, audio

    def __len__(self):
        return len(self.raw_data)

    def shuffle_list(self, list):
        random.shuffle(list)


class NextQADataset(Data.Dataset):
    def __init__(self, args, data_path, map_path, split, video_path):
        super(NextQADataset, self).__init__()
        self.tokenizer = Tokenizer(model_path= args.llama_model_path + '8B/tokenizer.model')
        self.max_words = args.max_seq_len
        self.data_path = Path(data_path)
        self.map = json.load(open(map_path,'r'))
        self.split = split
        self.video_path = Path(video_path)
        self.raw_data = pd.read_csv(data_path)
        print(f"number of examples in split {split}: {len(self.raw_data)}\n")
        self.preprocess()
    
    def preprocess(self):
        image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        feature_directory = self.video_path/"processed_features"
        feature_directory.mkdir(exist_ok=True)
        for _, row in self.raw_data.iterrows():
            id = str(row['video'])
            if not (feature_directory/(id+"_video.pt")).exists():
                video_file_path = self.video_path/(self.map[id] + ".mp4")
                audio_file_path = self.video_path/(self.map[id] + ".wav")
                command = f'/ocean/projects/cis240055p/skhyalia/NExT-QA/extract_or_generate_audio.sh {video_file_path} {audio_file_path}'
                os.system(command)
                container = av.open(video_file_path)
                indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
                video = read_video_pyav(container=container, indices=indices)
                video = image_processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
                waveform, sampling_rate = soundfile.read(audio_file_path)
                audio = feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.squeeze(0)
                torch.save(video, feature_directory/(id+"_video.pt"))
                torch.save(audio, feature_directory/(id+"_audio.pt"))
                gc.collect()
        return
    
    def tokenize(self,prompt,answer):
        example=prompt + ' ' + answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask
    
    def __getitem__(self, idx):
        data_point = self.raw_data.iloc[idx]
        id = str(data_point['video'])
        video = torch.load(self.video_path/"processed_features"/(id+"_video.pt"))
        audio = torch.load(self.video_path/"processed_features"/(id+"_audio.pt"))
        prompt_text = f"You will be given a question about a video and five possible answer options.\nQuestion: {data_point['question']}?\nPossible Answer Choices:\n(1) {data_point['a0']}\n(2) {data_point['a1']}\n(3) {data_point['a2']}\n(4) {data_point['a3']}\n(5) {data_point['a4']}\nOutput the final answer in the format \"Final Answer: (X) Y.\" where X is the correct digit choice and Y is the text of the choice. Never say \"unknown\" or \"unsure\", or \"None\", instead provide your most likely guess.\n"
        prompt_text+="Final Answer: "
        prompt_text='\n'+prompt_text
        prompt_text = prompt_text.replace("  ", " ").strip()
        answer_tag = f"a{data_point['answer']}"
        prompt_answer = f"({data_point['answer'] + 1}) {data_point[answer_tag]}."
        example, labels, example_mask, label_mask=self.tokenize(prompt_text,prompt_answer)
        gc.collect()
        return example, labels, example_mask, video, audio

    def __len__(self):
        return len(self.raw_data)

    def shuffle_list(self, list):
        random.shuffle(list)
