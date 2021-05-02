# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os

import tensorflow as tf
import pandas as pd

from sklearn.metrics import roc_auc_score

from data_load import get_batch
from model import Transformer, S2S, S2T
from hparams import Hparams
from utils import *
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp)


logging.info("# Prepare test batches")

test_batches, num_test_batches, _,  _, df = get_batch(hp.caption_path, hp.video_path, hp.n_video, hp.n_global, hp.n_action, hp.n_interaction, hp.batch_size, hp.types, False, "test")

iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")

ixtoword = pd.Series(np.load('./ixtowordGAI.npy', allow_pickle=True).tolist())
wordtoix = pd.Series(np.load('./wordtoixGAI.npy', allow_pickle=True).tolist())

hp.word = len(wordtoix)

#m = Transformer(hp, wordtoix, ixtoword)
m = S2S(hp, wordtoix, ixtoword)
#m = Transformer(hp)
y_hat, vp = m.eval(xs, ys)

logging.info("# Session")
with tf.Session() as sess:
    saver = tf.train.Saver()


    ckpt = os.path.join(os.path.join(os.path.join(hp.save_dir, "best"), hp.log_dir), "best")

    saver.restore(sess, ckpt)

    sess.run(test_init_op)

    logging.info("# get hypotheses")

    hypotheses, path, ggs, gas, gis= get_hypotheses(num_test_batches, sess, y_hat,vp, m.idx2token)

    logging.info("# write results")
    model_output = ckpt.split("/")[-1]

    logging.info("# calc bleu score and append it to translation")

    bg, ba, bi = calculate_bleu(df, path, ggs, gas, gis)

    label_i, hypothesis_i = calculate_types(df,path, gis)
    auc = roc_auc_score(label_i, hypothesis_i)

    print(bg, ba, bi, auc)

    logging.info("# write results")
    model_output = "Test_Result"
    if not os.path.exists(os.path.join(hp.save_dir,hp.test_dir)): os.makedirs(os.path.join(hp.save_dir, hp.test_dir))
    translation = os.path.join(os.path.join(hp.save_dir, hp.test_dir), model_output)
    with open(translation, 'w') as fout:
        path = [x.decode('utf-8') for x in path]
        for p, h in zip(path, hypotheses):
            fout.write(p + "\t" + h + "\n")


