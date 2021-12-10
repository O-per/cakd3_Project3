import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import sentencepiece as spm
import sqlite3


def get_data(text_path):
  con_ = sqlite3.connect(text_path)
  cur = con_.cursor()
  cur.execute("select article from abc order by random()")
  random = cur.fetchall() 
  return random

def get_txt(random1):
  #난수 생성
  i = np.random.randint(len(random1))
  return random1[i][0]

#summary - 제공된 텍스트
def make_summary(source_text, tokenizer, model):
  device = torch.device('cpu')
  tokenized_text = tokenizer.encode(source_text, return_tensors="pt").to(device)
  # summmarize 
  summary_ids = model.generate(tokenized_text,
                                      num_beams=2,
                                      no_repeat_ngram_size=3,
                                      min_length=30,
                                      max_length=250,
                                      repetition_penalty=2.5,
                                      length_penalty=1.0,
                                      early_stopping=True)

  summary_txt = tokenizer.decode(summary_ids[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

  return summary_txt

def b_score(input_text, sum_txt):
  import bert_score 
  cand = [f'{input_text}']
  ref = [f'{sum_txt}']
  P,R,F = bert_score.score(cands = cand, refs = ref, lang='ko', device='cpu') 

  b_score = np.round((float(F[0])+float(P[0])+float(R[0]))/3,3)
  return b_score

def re_text(sum_txt):
  sum_txt = sum_txt.replace('<extra_id_0> ','')
  sum_txt = sum_txt.replace('summarize: ','')
  sum_txt = sum_txt.replace('summaryze: ','')
  return sum_txt


def scored(user_text, sum_txt):
  score = b_score(user_text, sum_txt)
  if score > 0.9:
    score_text = 'Perfect'
  elif score > 0.7:
    score_text = 'Great'
  elif score > 0.5:
    score_text = 'Good'
  else:
    score_text = 'Try again'
  return score_text