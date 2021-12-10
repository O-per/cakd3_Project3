# -*- coding: utf-8 -*-

## 코랩에서 돌릴때 코드(__init__.py)
# 2021.12.08

from flask import Flask, g, request, url_for, Response, redirect, make_response, render_template
from flask_ngrok import run_with_ngrok
from text_1 import get_txt, make_summary, b_score, re_text, scored, get_data


app = Flask(__name__)
app.debug = True  # use only debug!!
run_with_ngrok(app) # 코랩 실행 시
app.static_folder = 'static' # 필요할지 모르겠음

# data
path1 = 'flaskapp/data/df1.db'
path2 = 'flaskapp/data/df2.db'
path3 = 'flaskapp/data/df3.db'
path4 = 'flaskapp/data/df4.db'
path5 = 'flaskapp/data/df5.db'

global con1, con2, con3, con4, con5
con1 = get_data(path1)
con2 = get_data(path2)
con3 = get_data(path3)
con4 = get_data(path4)
con5 = get_data(path5)

#-------------------------------
from transformers import T5Tokenizer
import torch
path = 'flaskapp/weight'
model = torch.load(path+'/model.pt', map_location='cpu') 
tokenizer = T5Tokenizer.from_pretrained(path)

import pandas as pd
#-------------------------------

@app.route("/")
def helloworld():
    return render_template('index.html', df='text')

@app.route('/home')
def about():
  return render_template('chatbot.html', df="text")


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        return redirect(url_for('index'))

    return render_template('settings.html')


@app.route('/user', methods=['GET', 'POST'])
def user():
    if request.method == 'POST':
        return redirect(url_for('index'))

    return render_template('user.html')

@app.route('/settings/us', methods=['GET', 'POST'])
def us():
    if request.method == 'POST':
        return redirect(url_for('us'))
    return render_template('settings1.html')

#------------------------------

@app.route('/summary/<int:num>', methods=['POST', 'GET'])
def summary_num(num=None, score_text=None, sum_txt=None):
  global data_text

  if num==1:
    data_text = get_txt(con1)
  elif num==2:
    data_text = get_txt(con2)  
  elif num==3:
    data_text = get_txt(con3)  
  elif num==4:
    data_text = get_txt(con4)  
  elif num==5:
    data_text = get_txt(con5) 

  return render_template('chatbot.html', data_text = data_text, num=num, ai_data=sum_txt, score_text=score_text)

@app.route('/summary/do', methods=['GET', 'POST'])
def summary1():
  global text
  user_text = request.form.get('user_text_text')
  sum_txt = make_summary(data_text, tokenizer, model)
  sum_txt = re_text(sum_txt)
  score_text = scored(user_text, sum_txt)  
  return render_template('chatbot1.html', data_text = data_text, ai_data=sum_txt, score_text=score_text, user_text=user_text, user_text_len = len(user_text))
 


#-------------------------------


@app.route('/topic', methods=['GET', 'POST'])
def topic():
    if request.method == 'POST':
        return redirect(url_for('index'))

    return render_template('topic1_index.html')


@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/stapup_index', methods=['GET', 'POST'])
def stapup_index():
    if request.method == 'POST':
        return redirect(url_for('index'))

    return render_template('stapup_index.html')

############################################################################
#---------------------------------------------------------------------------
# ../helloflask.py
from flaskapp import app

app.run()