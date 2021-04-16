#-*- coding: utf-8 -*-
from hparams import Hparams
import logging
from utils import *
import argparse
from data_load import get_batch
from model import Transformer, S2S, S2T

import tracemalloc

import math

from tqdm import tqdm

import gc


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import random
import csv


logging.basicConfig(level=logging.INFO)

logging.info("# Hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp)

train_batches, num_train_batches, num_train_samples, wordtoix, ixtoword, _ = get_batch(hp.caption_path, hp.video_path, hp.n_video, hp.n_global, hp.n_action, hp.n_interaction, hp.batch_size, hp.types, True, "train")

eval_batches, num_eval_batches, num_eval_samples, _,  _, df = get_batch(hp.caption_path, hp.video_path, hp.n_video, hp.n_global, hp.n_action, hp.n_interaction, hp.batch_size, hp.types, False, "valid")


iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)

xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load Model")
m = S2T(hp, wordtoix, ixtoword)
#m = Transformer(hp, wordtoix, ixtoword)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat, vp= m.eval(xs, ys)


logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)

with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.log_dir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.log_dir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.log_dir, sess.graph)

    sess.run(train_init_op)
    print(num_train_batches)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)

    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            logging.info("# test evaluation")
            _, = sess.run([eval_init_op])
            #summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses, ggs, gas, gis= get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat,m.idx2token)

            logging.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.eval_dir): os.makedirs(hp.eval_dir)
            translation = os.path.join(hp.eval_dir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and append it to translation")

            path = sess.run(vp)
            bg, ba, bi = calculate_bleu(df, path, ggs, gas, gis)
            del path
            gc.collect()

            with open(translation, "a") as fout:
                fout.write("step {}  G : {} \t A : {} \t I : {} \n".format(_gs, bg, ba, bi))
            logging.info("step {}  G : {} \t A : {} \t I : {} \n".format(_gs, bg, ba, bi))

            logging.info("# save models")
            ckpt_name = os.path.join(hp.log_dir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)

    summary_writer.close()
