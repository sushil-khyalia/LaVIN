# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from lavin.tokenizer import Tokenizer
from lavin.eval_model import Transformer

class LaVIN_Generator:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def insert_audio_video_embeds(self, examples, labels, video_embeds, prefix_video, audio_embeds, prefix_audio):
        _bsz, seqlen,_ = examples.shape
        new_examples=[]
        new_labels=[]
        for i, (example,label) in enumerate(zip(examples,labels)):
            new_example=torch.cat([example[:1],prefix_video,video_embeds[i],prefix_audio,audio_embeds[i],example[1:]],0)
            new_label=torch.cat([label[:1],
                                    torch.zeros(prefix_video.shape[0]+video_embeds.shape[1]+prefix_audio.shape[0]+audio_embeds.shape[1]).to(examples.device).type_as(labels),
                                    label[1:]])
            new_example = new_example[:seqlen]
            new_label = new_label[:seqlen]
            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)    
        return new_examples, new_labels

    
    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        videos: torch.Tensor,
        audios: torch.Tensor,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        self.model.eval()

        prefix_audio_token = self.tokenizer.encode("Audio: ", bos=False, eos=False)
        prefix_video_token = self.tokenizer.encode("Video: ", bos=False, eos=False)
        
        audios=audios.cuda()
        videos=videos.cuda()
        self.model.video_backbone.cuda()
        self.model.audio_backbone.cuda()

        
        video_embeds = self.model.video_backbone(videos)
        audio_embeds = self.model.audio_backbone(audios)
        video_embeds=self.model.video_adapter_proj(video_embeds)
        audio_embeds=self.model.audio_adapter_proj(audio_embeds)

        n_feats_video = 12
        n_feats_audio = 32
        prompt_tokens=[]
        for i,x in enumerate(prompts):
            token_idx=prefix_video_token+[0]*n_feats_video+prefix_audio_token+[0]*n_feats_audio+self.tokenizer.encode(x, bos=False, eos=False)
            prompt_tokens.append(token_idx)


        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        input_text_mask=torch.zeros_like(tokens).bool()

        for k, t in enumerate(prompt_tokens):
            t=t[:total_len]
            tokens[k, : len(t)] = torch.tensor(t).long()
            input_text_mask[k,:len(t)]=True

        token_embeds=self.model.tok_embeddings(tokens)
        modality_embedding=self.model.adapter_modality_embedding(torch.ones(bsz,1).long().cuda())

        for i in range(len(token_embeds)):
            pos_video=len(prefix_video_token)
            pos_audio=len(prefix_video_token) + n_feats_video + len(prefix_audio_token)
            #insert image emebedding into the sequence
            token_embed=torch.cat([token_embeds[i,:pos_video],video_embeds[i],token_embeds[i,pos_video+n_feats_video:pos_audio],audio_embeds[i],token_embeds[i,pos_audio+n_feats_audio:]],0)
            token_embeds[i]=token_embed



        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):

            if prev_pos==0:
                h=torch.cat([modality_embedding,token_embeds[:,prev_pos:cur_pos]],1)
            else:
                h=token_embeds[:,prev_pos:cur_pos]
            logits = self.model.forward(h, prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated

            next_token_embeds = torch.where(
                input_text_mask[:, cur_pos,None], token_embeds[:, cur_pos], self.model.tok_embeddings(next_token)
            )
            token_embeds[:,cur_pos]=next_token_embeds

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))


        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
