import json
import os
import numpy as np
import nltk

nltk.download('wordnet')

import tensorflow as tf

import logging

import gc

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


def get_hypotheses(num_batches, sess, tensor,vp,dict):
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
    vps = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        v = sess.run(vp)
        vps.extend(v.tolist())
        hypotheses.extend(h.tolist())
    _hypotheses, ggs, gas, gis = postprocess(hypotheses, dict)

    return _hypotheses, vps, ggs, gas, gis


def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    ggs = []
    gas = []
    gis = []
    for h in hypotheses:
        sent = []
        for idx in h:
            if idx >= len(idx2token):
                idx=7
            sent.append(idx2token[idx])

        sent = np.array(sent)
        punc_gb=np.argmax(sent=='<gbos>')+1
        punc_ge =np.argmax(sent=='<geos>')
        punc_ab =np.argmax(sent=='<abos>')+1
        punc_ae =np.argmax(sent=='<aeos>')
        punc_ib =np.argmax(sent=='<ibos>')+1
        punc_ie =np.argmax(sent=='<ieos>')

        gg = sent[punc_gb:punc_ge]
        ga = sent[punc_ab:punc_ae]
        gi = sent[punc_ib:punc_ie]

        gg = ' '.join(gg)
        ga = ' '.join(ga)
        gi = ' '.join(gi)

        ggs.append(gg)
        gas.append(ga)
        gis.append(gi)
        _hypotheses.append(gg+"\t"+ga+"\t"+gi)
    return _hypotheses, ggs, gas, gis

def calculate_bleu(df, vp, ggs, gas, gis):
    bleu_global = 0
    bleu_action = 0
    bleu_interaction = 0


    dvp = []
    dgis = []
    dgas = []
    dggs = []
    for vn, gg, ga, gi in zip(vp, ggs, gas, gis):
        if vn not in dvp:
            dvp.append(vn)
            dggs.append(gg)
            dgas.append(ga)
            dgis.append(gi)


    for vn, gg, ga, gi in zip(dvp, dggs, dgas, dgis):
        ref = df[df['video_path']==vn.decode('utf-8')]

        refg = ref.drop_duplicates(['Global'], keep='first')
        refa = ref.drop_duplicates(['Action'], keep='first')
        refi = ref.drop_duplicates(['Interaction'], keep='first')
        ref_global = refg['Global'].values.tolist()
        ref_action = refa['Action'].values.tolist()
        ref_interaction = refi['Interaction'].values.tolist()

        bleu_global += nltk.translate.bleu_score.sentence_bleu(ref_global, gg)
        bleu_action += nltk.translate.bleu_score.sentence_bleu(ref_action, ga)
        bleu_interaction += nltk.translate.bleu_score.sentence_bleu(ref_interaction, gi)

        del refg
        del refa
        del refi

        gc.collect()

    return bleu_global/len(dggs), bleu_action/len(dgas), bleu_interaction/len(dgis)

def calculate_types(df, vp, gis):
    dvp = []
    dgis = []
    for vn, gi in zip(vp, gis):
        if vn not in dvp:
            dvp.append(vn)
            dgis.append(gi)

    y_hat = []
    y = []
    for vn, gi in zip(dvp, dgis):
        ref = df[df['video_path']==vn.decode('utf-8')]
        refi = ref.drop_duplicates(['video_path'], keep='first')
        ref_interaction = refi['Interaction'].values

        if ref_interaction=="<NONE>":
            y.append(1)
        else:
            y.append(0)

        if gi=="<none>":
            y_hat.append(1)
        else:
            y_hat.append(0)
    return y, y_hat



def load_hparams(hp):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''

    path = os.path.join(hp.save_dir, hp.log_dir)
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        hp.f = v


