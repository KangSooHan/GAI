#-*- coding: utf-8 -*-
from hparams import Hparams
import logging
from utils import *
import argparse
from data_load import get_batch
from model import Transformer

import math

from tqdm import tqdm


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import random
import csv


def train(hp, train_df, valid_df):
    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []
    test_output = open('output.txt', 'w')

    for epoch in range(0, hp.epochs):
        loss_to_draw_epoch = []
        index = list(train_df.index)
        np.random.shuffle(index)
        train_data = train_datas.loc[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)

        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            start_time = time.time()

            current_batch = train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_caption_matrix = tf.keras.preprocessing.sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption)
            current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)


            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))

            nonzeros = np.array(list(map(lambda x: (x!=0).sum() + 1, current_caption_matrix)))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]-1] = 1

            for a, caption in enumerate(current_caption_ind):
                for b, word in enumerate(caption):
                    if word <8:
                        current_caption_masks[a, b]*=0.5


            _, loss_val, probs_val = sess.run(
                    [train_op, tf_loss, tf_probs],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask: current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            loss_to_draw_epoch.append(loss_val)


            print('idx:', start, 'Epoch:', epoch, 'loss:', loss_val, 'Elapsed time:', str((time.time()-start_time)))
            loss_fd.write('epoch ' + str(epoch) + 'loss ' + str(loss_val) + '\n')

        if np.mod(epoch, 10) == 0 and epoch > 0:
            print("Epoch ", epoch, "is done.")

        if np.mod(epoch, 10) == 0 and epoch >= 0:
            generated_sentences = []
            for bs in probs_val:
                generated_words = []
                for prob in bs:
                    generated_words.append(ixtoword[prob])
                generated_sentences.append(generated_words)



            ggs=[]
            gas=[]
            gis=[]

            for gs in generated_sentences:
                punc_gb =np.argmax(np.array(gs)=='<gbos>')+1
                punc_ge =np.argmax(np.array(gs)=='<geos>')+1
                punc_ab =np.argmax(np.array(gs)=='<abos>')+1
                punc_ae =np.argmax(np.array(gs)=='<aeos>')+1
                punc_ib =np.argmax(np.array(gs)=='<ibos>')+1
                punc_ie =np.argmax(np.array(gs)=='<ieos>')+1

                gg = gs[punc_gb:punc_ge]
                ga = gs[punc_ab:punc_ae]
                gi = gs[punc_ib:punc_ie]

                ggs.append(' '.join(gg))
                gas.append(' '.join(ga))
                gis.append(' '.join(gi))


            bleu_g, bleu_a, bleu_i =calculate_bleu(train_datas,current_batch['video_path'], ggs, gas, gis, current_captions)

            print(ggs[0],"\t", bleu_g,"\n", gas[0], "\t",bleu_a,"\n", gis[0],"\t", bleu_i)

            #generated_sentence = generated_sentence.replace('<bos> ', '')
            #generated_sentence = generated_sentence.replace(' <eos>', '')

            #print(generated_sentence)


            print("Epoch ", epoch, "is done. Saving the model...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
    loss_fd.close()


logging.basicConfig(level=logging.INFO)

logging.info("# Hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp)

train_batches, num_train_batches, num_train_samples, wordtoix, ixtoword = get_batch(hp.caption_path, hp.video_path, hp.n_video, hp.n_global, hp.n_action, hp.n_interaction, hp.batch_size, hp.types, True, "train")

eval_batches, num_eval_batches, num_eval_samples, _, _= get_batch(hp.caption_path, hp.video_path, hp.n_video, hp.n_global, hp.n_action, hp.n_interaction, hp.batch_size, hp.types, False, "valid")

iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)

xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load Model")
m = Transformer(hp, wordtoix, ixtoword)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat= m.eval(xs, ys)

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
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            logging.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.eval_dir): os.makedirs(hp.eval_dir)
            translation = os.path.join(hp.eval_dir, model_output)
            with open(translation, 'w') as fout:
                print(hypotheses)
                #fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and append it to translation")
            #calc_bleu(hp.eval3, translation)

            logging.info("# save models")
            ckpt_name = os.path.join(hp.log_dir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()
