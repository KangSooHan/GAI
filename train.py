#-*- coding: utf-8 -*-
from hparams import Hparams
import logging
from utils import *
import argparse
from data_load import get_batch
from model import Transformer, S2S, S2T

import shutil
from sklearn.metrics import roc_auc_score

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

hp.caption=0
if 'G' in hp.types :
    hp.caption += hp.n_global

if 'A' in hp.types :
    hp.caption += hp.n_action

if 'I' in hp.types :
    hp.caption += hp.n_interaction
save_hparams(hp)

train_batches, num_train_batches, wordtoix, ixtoword, _ = get_batch(hp.caption_path, hp.video_path, hp.n_video, hp.n_global, hp.n_action, hp.n_interaction, hp.batch_size, hp, True, "train")

hp.word = len(ixtoword)

eval_batches, num_eval_batches, _,  _, df = get_batch(hp.caption_path, hp.video_path, hp.n_video, hp.n_global, hp.n_action, hp.n_interaction, hp.batch_size, hp, False, "valid")


iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)

xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load Model")
#m = S2S(hp, wordtoix, ixtoword)
#m = S2T(hp, wordtoix, ixtoword)
m = Transformer(hp, wordtoix, ixtoword)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat, vp= m.eval(xs, ys)


logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)

with tf.Session() as sess:
    if not os.path.exists(os.path.join(hp.save_dir, hp.chkpt_dir)): os.makedirs(os.path.join(hp.save_dir, hp.chkpt_dir))
    ckpt = tf.train.latest_checkpoint(os.path.join(hp.save_dir, hp.chkpt_dir))
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(os.path.join(hp.save_dir, hp.log_dir)): os.makedirs(os.path.join(hp.save_dir, hp.log_dir))
        save_variable_specs(os.path.join(os.path.join(hp.save_dir, hp.log_dir), "specs"))
    else:
        saver.restore(sess, ckpt)


    summary_writer = tf.summary.FileWriter(os.path.join(hp.save_dir, hp.log_dir), sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)

    max_score=0;

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
            hypotheses,path, ggs, gas, gis= get_hypotheses(num_eval_batches, sess, y_hat,vp,m.idx2token)

            logging.info("# calc bleu score and append it to translation")

            bg, ba, bi = calculate_bleu(df, path, ggs, gas, gis)

            label_i, hypothesis_i = calculate_types(df,path, gis)
            auc = roc_auc_score(label_i, hypothesis_i)

            logging.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(os.path.join(hp.save_dir,hp.eval_dir)): os.makedirs(os.path.join(hp.save_dir, hp.eval_dir))
            translation = os.path.join(os.path.join(hp.save_dir, hp.eval_dir), model_output)
            with open(translation, 'w') as fout:
                path = [x.decode('utf-8') for x in path]
                for p, h in zip(path, hypotheses):
                    fout.write(p + "\t" + h + "\n")
                    #fout.write("\n".join(path) + "\t".join(hypotheses))

            with open(translation, "a") as fout:
                fout.write("\nstep {}  G : {} \t A : {} \t I : {} \t I(AUC) : {} \n".format(_gs, bg, ba, bi, auc))
            logging.info("step {}  G : {} \t A : {} \t I : {} \t I(AUC) : {}\n".format(_gs, bg, ba, bi, auc))


            logging.info("# save models")
            ckpt_name = os.path.join(os.path.join(hp.save_dir, hp.log_dir), model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# save best models")
            now_score = bg+ba+bi/10+auc/10
            if(max_score < now_score):
                max_score = now_score

                if os.path.exists(os.path.join(os.path.join(hp.save_dir, "best"), hp.eval_dir)):
                    shutil.rmtree(os.path.join(os.path.join(hp.save_dir, "best"), hp.eval_dir))
                os.makedirs(os.path.join(os.path.join(hp.save_dir, "best"), hp.eval_dir))


                if os.path.exists(os.path.join(os.path.join(hp.save_dir, "best"), hp.log_dir)):
                    shutil.rmtree(os.path.join(os.path.join(hp.save_dir, "best"), hp.log_dir))

                os.makedirs(os.path.join(os.path.join(hp.save_dir, "best"), hp.log_dir))
                ckpt_name = os.path.join(os.path.join(os.path.join(hp.save_dir, "best"), hp.log_dir), "best")
                saver.save(sess, ckpt_name, global_step=_gs)

                model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)

                translation = os.path.join(os.path.join(os.path.join(hp.save_dir, "best"), hp.eval_dir), model_output)
                with open(translation, 'w') as fout:
                    for p, h in zip(path, hypotheses):
                        fout.write(p + "\t" + h + "\n")

                with open(translation, "a") as fout:
                    fout.write("\nstep {}  G : {} \t A : {} \t I : {} \t I(AUC) : {} \n".format(_gs, bg, ba, bi, auc))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)

    summary_writer.close()
