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
import  json, re,random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from util.base_prompt import *
import torch
from lavin import Tokenizer
import copy
import pandas as pd
from util.misc import sample_frame_indices, read_video_pyav
from vivit import VivitImageProcessor
from whisper import WhisperFeatureExtractor
import av
import soundfile
import numpy as np
import gc

class MOSIDataset(Data.Dataset):
    def __init__(self, args, split, model_path, max_words=512):
        super(MOSIDataset, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------        
        self.tokenizer = Tokenizer(model_path= model_path + '8B/tokenizer.model')
        self.max_words = max_words
        self.split=split
        self.raw_data = pd.read_csv('/work/skhyalia/dataset_original/mosi_sentiment_%s.csv' % (split))
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
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        video = self.image_processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        audio_path = data_point['audio']
        waveform, sampling_rate = soundfile.read(audio_path)
        audio = self.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.squeeze(0)
        prompt_text = f"Text: {data_point['sentence']}"
        prompt_text+="Response: "
        prompt_text='\n'+prompt_text
        prompt_text = prompt_text.replace("  ", " ").strip()
        label = str(np.round(data_point['y']))
        prompt_answer = f"The sentiment is {label}"
        example, labels, example_mask, label_mask=self.tokenize(prompt_text,prompt_answer)

        return example, labels, example_mask, video, audio

    def __len__(self):
        return len(self.raw_data)

    def shuffle_list(self, list):
        random.shuffle(list)

class MOSIDatasetForClassification(Data.Dataset):
    def __init__(self, args, split, model_path, max_words=512):
        super(MOSIDatasetForClassification, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------        
        self.tokenizer = Tokenizer(model_path= model_path + '8B/tokenizer.model')
        self.max_words = max_words
        self.split=split
        self.raw_data = pd.read_csv('/work/skhyalia/dataset_original/mosi_sentiment_%s.csv' % (split))
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
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        video = self.image_processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        audio_path = data_point['audio']
        waveform, sampling_rate = soundfile.read(audio_path)
        audio = self.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.squeeze(0)
        prompt_text = f"Text: {data_point['sentence']}"
        prompt_text+="Response: "
        prompt_text='\n'+prompt_text
        prompt_text = prompt_text.replace("  ", " ").strip()
        prompt_answer = f"The sentiment is "
        example, labels, example_mask, label_mask=self.tokenize(prompt_text,prompt_answer)

        return example, labels, torch.tensor(np.round(data_point['y'])+3), example_mask, video, audio

    def __len__(self):
        return len(self.raw_data)

    def shuffle_list(self, list):
        random.shuffle(list)

class MOSIDatasetForRegression(Data.Dataset):
    def __init__(self, args, path, split, emotion, model_path, max_words=512):
        super(MOSIDatasetForRegression, self).__init__()
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
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
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


class ScienceQADataSet(Data.Dataset):
    def __init__(self, args,split,model_path,max_words=512,max_image_feats=1):
        super(ScienceQADataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
        pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
        captions = json.load(open(args.caption_file))["captions"]
        self.image_path=os.path.join(args.data_root,'images',split)
        self.tokenizer = Tokenizer(model_path= model_path + '8B/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else ""

        self.qids = pid_splits['%s' % (split)]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

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

        prompt_question,prompt_answer= build_prompt(self.problems,self.qids[idx],self.args)
        answer,choices,qid=self.problems[self.qids[idx]]["answer"], self.problems[self.qids[idx]]["choices"],self.qids[idx]

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            image_mask=torch.cat([torch.Tensor([float('-inf')]*self.max_image_feats),torch.zeros(self.max_words)])
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            image_mask=torch.zeros(self.max_words+self.max_image_feats)
            indicator=0

        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image,indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)



class InstrcutDataSet(Data.Dataset):
    def __init__(self, args,split,model_path,max_words=512,max_image_feats=1):
        super(InstrcutDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.data = json.load(open(os.path.join(args.data_root, 'all_data.json')))[split]

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        self.qids = [item['qid'] for item in self.data]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer,max_words=512):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = max_words - example.shape[0]
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

        prompt_question=self.data[idx]['instruction']
        prompt_answer=self.data[idx]['answer']

        if self.data[idx]['image'] is not None:
            # image_path='../data/images/train' if self.data[idx]['image_source']=='sqa' else '../data/images/train2014'
            if self.data[idx]['image_source'] == 'sqa':
                image = Image.open(os.path.join('../data/images/train', self.qids[idx], 'image.png')).convert('RGB')
            else:
                image = Image.open(os.path.join('../data/images/train2014',   'COCO_train2014_'+self.data[idx]['image'])).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            indicator=0

        # print(prompt_question,prompt_answer)
        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image,indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.options = ["A", "B", "C", "D", "E"]
            self.use_caption = True
            self.prompt_format = 'CQM-A'
            self.data_root = './data'
            self.output_root = './output'
            self.caption_file = './data/captions.json'
    cfg=Cfg()
    dataset=ScienceQADataSet(cfg,'val','./data/weights')
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True)
    max_question_len=0
    max_answer_len=0
    #406 max question
    for prompt_questions,question_mask,images,image_masks,prompt_answers,answers,qids in data_loader:
        print(prompt_questions)
        print(answers)
    #     if len(prompt_questions[0].split())>max_question_len:
    #         max_question_len=len(prompt_questions[0].split())
    #     if len(prompt_answers[0].split())>max_answer_len:
    #         max_answer_len=len(prompt_answers[0].split())
    # print(max_question_len,max_answer_len)






