import json
import os
import numpy as np
import nltk
import tensorflow as tf

import logging

logging.basicConfig(level=logging.INFO)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)



def calc_num_batches(total_num, batch_size):
    return total_num // batch_size + (total_num % batch_size !=0)

def save_hparams(hparams):
    '''Save hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    '''

    path = os.path.join(hparams.save_dir, hparams.log_dir)
    os.makedirs(path, exist_ok=True)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def calc_len(hp):
    caplen=0
    if 'G' in hp.types:
        caplen += hp.n_global
    if 'A' in hp.types:
        caplen += hp.n_action
    if 'I' in hp.types:
        caplen += hp.n_interaction

    return hp.n_video, caplen

def calculate_bleu(df,video_name, ggs, gas, gis, cap):

    bleu_global = 0
    bleu_action = 0
    bleu_interaction = 0
    for vn, gg, ga, gi, ca in zip(video_name, ggs, gas, gis, cap):
        ref = df[df['video_path']==vn]

        refg = ref.drop_duplicates(['Global'], keep='first')
        refa = ref.drop_duplicates(['Action'], keep='first')
        refi = ref.drop_duplicates(['Interaction'], keep='first')
        ref_global = refg['Global'].values.tolist()
        ref_action = refa['Action'].values.tolist()
        ref_interaction = refi['Interaction'].values.tolist()


        bleu_global += nltk.translate.bleu_score.sentence_bleu(ref_global, gg)
        bleu_action += nltk.translate.bleu_score.sentence_bleu(ref_action, ga)
        bleu_interaction += nltk.translate.bleu_score.sentence_bleu(ref_interaction, gi)

        #print(cap)

    return bleu_global/len(ggs), bleu_action/len(ggs), bleu_interaction/len(ggs)

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")


def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    '''
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    #hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples]


