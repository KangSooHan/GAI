#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import sys
import time
import cv2
#from keras.preprocessing import sequence
import pdb
import random
import nltk

from sklearn import metrics


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_video, n_caption, bias_init_vector=None, attention=None, dropout_prob=1.0):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_video = n_video
        self.n_caption = n_caption
        self.attention = attention


        self.w_enc_out =  tf.Variable(tf.random_uniform([dim_hidden, dim_hidden]), dtype=tf.float32, name='w_enc_out')
        self.w_dec_state =  tf.Variable(tf.random_uniform([dim_hidden, dim_hidden]), dtype=tf.float32, name='w_dec_state')
        self.v = tf.Variable(tf.random_uniform([dim_hidden, 1]), dtype=tf.float32, name='v')


        self.Wemb = tf.Variable(tf.random_uniform([n_words, 100], -0.1, 0.1), name='Wemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)

        self.lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=dropout_prob)
        self.lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=dropout_prob)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1,0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')


        self.video = tf.placeholder(tf.float32, [None, self.n_video, self.dim_image])
        self.video_mask = tf.placeholder(tf.int32, [None])
        self.caption = tf.placeholder(tf.int32, [None, self.n_caption+1])
        self.caption_mask = tf.placeholder(tf.float32, [None, self.n_caption+1])
        self.video_pad = tf.zeros([self.batch_size, n_video, 100])
        self.cap_pad = tf.zeros([self.batch_size, n_caption, self.dim_hidden])
        self.video_pad1 = tf.zeros([self.batch_size, n_video, self.dim_hidden])

    def build_model(self):


        video_flat = tf.reshape(self.video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [-1, self.n_video, self.dim_hidden])

        prob = []
        loss = 0.0

        ###### encoding
        with tf.variable_scope("encoding_vid", reuse=tf.AUTO_REUSE) as scope:
            output1, state1 = tf.nn.dynamic_rnn(self.lstm1, image_emb,sequence_length=self.video_mask, dtype=tf.float32)

        with tf.variable_scope("encoding_cap", reuse=tf.AUTO_REUSE) as scope:
            output2, state2 = tf.nn.dynamic_rnn(self.lstm2, tf.concat([self.video_pad, output1],2), sequence_length=self.video_mask, dtype=tf.float32)


        with tf.variable_scope("decoding_vid", reuse=tf.AUTO_REUSE) as scope:
            output1, state1 = tf.nn.dynamic_rnn(self.lstm1, self.cap_pad, dtype=tf.float32)

        #if self.with_attention:
        def bahdanau_attention(time, prev_output=None):
            
            if time == 0:
                H_t = output1[:,-1, :] # encoder last output as first target input, H_t
            else:
                H_t = prev_output

            H_t = tf.matmul(H_t, self.w_dec_state)
            H_s = tf.identity(output1) # copy
                
            H_s = tf.reshape(H_s, (-1, self.dim_hidden))
            score = tf.matmul(H_s, self.w_enc_out)
            score = tf.reshape(score, (self.batch_size,-1, self.dim_hidden))
            score = tf.add(score, tf.expand_dims(H_t, 1))
            
            score = tf.reshape(score, (-1, self.dim_hidden))
            score = tf.matmul(tf.tanh(score), self.v)
            score = tf.reshape(score, (self.batch_size, 60, 1))
            score = tf.nn.softmax(score, dim=-1, name='alpha')

            H_s = tf.reshape(H_s, (self.batch_size,-1, self.dim_hidden))
            C_i = tf.reduce_sum(tf.multiply(H_s, score), axis=1)
            print(C_i)
            return C_i


        def luong_attention(time, prev_output=None):
            
            if time == 0:
                H_t = h_src[-1,:, :] # encoder last output as first target input, H_t
            else:
                H_t = prev_output
            H_s = tf.identity(h_src) # copy

            H_s = tf.reshape(H_s, (-1, self.dim_hidden))
            score = tf.matmul(H_s, weights['w_enc_out'])

            H_s = tf.reshape(H_s, (-1, batch_size, self.dim_hidden))
            score = tf.reshape(score, (-1, batch_size, self.dim_hidden))
            
            score = tf.reduce_sum(tf.multiply(score, tf.expand_dims(H_t, 0)), axis=-1)
            score = tf.nn.softmax(score, dim=-1, name='alpha')

            C_i = tf.reduce_sum(tf.multiply(H_s, tf.expand_dims(score, 2)), axis=0)
            return C_i

        with tf.variable_scope("decoding_cap", reuse=tf.AUTO_REUSE) as scope:
            for i in range(self.n_caption):
                current_embed = tf.nn.embedding_lookup(self.Wemb, self.caption[:,i])

                if i==0:
                    #C_i = bahdanau_attention(i)
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size], dtype=tf.int32))
                else:
                    #C_i = bahdanau_attention(i, state2[1])
                    current_embed = tf.nn.embedding_lookup(self.Wemb, [p])

                output2, state2 = self.lstm2(tf.concat([current_embed, output1[:,i,:]], 1), state2)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)

                p = tf.argmax(logit_words, 1)[0]

                prob.append(p)

        return self.video, self.video_mask, self.caption, self.caption_mask, prob


def get_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep='^')
    video_data['video_path'] = video_data.apply(lambda row: row['Video']+'-{:06d}'.format(int(row['Vid_num'])) + '_' + row['Type'] +'.npy', axis=1)
    print(video_data)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Global'].map(lambda x: isinstance(x, str))]
    video_data = video_data[video_data['Action'].map(lambda x: isinstance(x, str))]
    video_data = video_data[video_data['Interaction'].map(lambda x: isinstance(x, str))]

    data = []

    def duple(df):
        copy = df.copy()
        for x in copy["Global"].unique():
            for y in copy["Action"].unique():
                for z in copy["Interaction"].unique():
                   data.append({"Global":x,
                                "Action":y,
                                "Interaction":z,
                                "video_path": str(copy["video_path"].unique())[2:-2]})

    video_data.groupby(['video_path']).apply(duple)

    new_data = pd.DataFrame(data)

    new_data['Global'] = new_data['Global'].map(lambda x: x.replace('.', '').replace(',','').replace('"','').replace('\n','').replace('?','').replace('!','').replace('\\','').replace('/',''))
    new_data['Action'] = new_data['Action'].map(lambda x: x.replace('.', '').replace(',','').replace('"','').replace('\n','').replace('?','').replace('!','').replace('\\','').replace('/',''))
    new_data['Interaction'] = new_data['Interaction'].map(lambda x: x.replace('.', '').replace(',','').replace('"','').replace('\n','').replace('?','').replace('!','').replace('\\','').replace('/',''))

    unique_filenames = new_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    random.seed(448)
    random.shuffle(unique_filenames)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = new_data[new_data['video_path'].map(lambda x: x in train_vids)]
    test_data = new_data[new_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print('preprocessing word counts and creating vocab based on word count threshold %d' % word_count_threshold)
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1


    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<gbos>'
    ixtoword[2] = '<geos>'
    ixtoword[3] = '<abos>'
    ixtoword[4] = '<aeos>'
    ixtoword[5] = '<ibos>'
    ixtoword[6] = '<ieos>'
    ixtoword[7] = '<unk>'
    ixtoword[8] = '<start>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<gbos>'] = 1
    wordtoix['<geos>'] = 2
    wordtoix['<abos>'] = 3
    wordtoix['<aeos>'] = 4
    wordtoix['<ibos>'] = 5
    wordtoix['<ieos>'] = 6
    wordtoix['<unk>'] = 7
    wordtoix['<start>'] = 8

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 9
        ixtoword[idx+9] = w

    word_counts['<pad>'] = nsents
    word_counts['<gbos>'] = nsents
    word_counts['<geos>'] = nsents
    word_counts['<abos>'] = nsents
    word_counts['<aeos>'] = nsents
    word_counts['<ibos>'] = nsents
    word_counts['<ieos>'] = nsents
    word_counts['<unk>'] = nsents
    word_counts['<start>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range


    for i in range(9):
        word_counts.pop(ixtoword[i])

    chart_values = sorted(word_counts.items(), reverse=True, key=lambda x: x[1])

    chart_x = [i[0] for i in chart_values]
    chart_y = [i[1] for i in chart_values]

    up_chart_x = chart_x[:50]
    up_chart_y = chart_y[:50]

    #자르고싶으면
    #chart_y = [100 if i>100 else i for i in chart_y]

    fig = plt.figure(figsize=(20,10))
    rects = plt.bar(chart_x, chart_y)
    plt.xticks([])
    plt.savefig('word_freq.png')


    fig = plt.figure(figsize=(20,10))
    rects = plt.bar(up_chart_x, up_chart_y)
    plt.xticks([])
    plt.savefig('down_word_freq.png')
    print(up_chart_x)

    return wordtoix, ixtoword, bias_init_vector

def calculate_bleu(df,video_name, ggs, gas, gis, cap):

    bleu_global = 0
    bleu_action = 0
    bleu_interaction = 0
    for vn, gg, ga, gi in zip(video_name, ggs, gas, gis):
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



def calculate_auc(df,video_name, gis, pred):
    for vn,gi in zip(video_name,gis):
        ref = df[df['video_path']==vn]
        refi = ref.drop_duplicates(['Interaction'], keep='first')
        ref_interaction = refi['Interaction'].values.tolist()

        if '<none>' in ref_interaction[0].lower():
            return 0 
        else:
            return 1




def test(model_path='./models_i/model-1700'):
    video_path = './data/UTE'
    video_feat_path = './rgb_feats'
    video_data_path = './data/UTE_caption.csv'

    train_data, test_data = get_video_data(video_data_path, video_feat_path, 0.9)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    test_videos = test_data["video_path"].unique()

    ixtoword = pd.Series(np.load('./data/ixtoword_i.npy', allow_pickle=True).tolist())

    bias_init_vector = np.load('./data/bias_init_vector_i.npy')

    print(len(ixtoword))


    #### train parameters
    dim_image = 4096
    dim_hidden = 1000
    n_video = 80
    n_gcaption = 30
    n_acaption = 20
    n_icaption = 20

    n_caption = n_gcaption + n_acaption + n_icaption
    n_epochs = 2000
    batch_size = 136
    learning_rate = 0.0001

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=1,
            n_video=n_video,
            n_caption=n_caption,)


    tf_video , tf_video_mask, tf_caption, tf_caption_mask,caption_tf = model.build_model()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open('S2VT_results.txt', 'w')
    test_output_cor_fd = open('S2VT_Description.txt', 'w')


    bgg=0
    bga=0
    bgi=0
    pred_i = []
    true_i = []
    for idx, video_feat_path in enumerate(test_videos):
        print(idx, video_feat_path)

        video_feat_val = np.load(video_feat_path)

        video_feat = np.zeros((n_video, dim_image))

        video_feat[:len(video_feat_val)] = video_feat_val

        video_masks = np.array(len(video_feat_val), dtype=np.int32)

        video_feat = np.expand_dims(video_feat, 0)
        video_masks = np.expand_dims(video_masks, 0)

        pad_caption = np.zeros([1, n_caption+1])
        pad_mask = np.zeros([1, n_caption+1])

        generated_word_index = sess.run(
                     caption_tf,
                    feed_dict={
                        tf_video:  video_feat,
                        tf_video_mask: video_masks,
                        tf_caption: pad_caption,
                        tf_caption_mask: pad_mask
                        })

        gs = []
        for prob in generated_word_index:
            gs.append(ixtoword[prob])



        ggs=[]
        gas=[]
        gis=[]

        #for gs in generated_sentences:
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


        if '<none>' in gis[0]:
            pred_i.append(0)
        else:
            pred_i.append(1)

        true = calculate_auc(test_data,[video_feat_path],gis, pred_i)
        true_i.append(true)

        bleu_g, bleu_a, bleu_i =calculate_bleu(test_data,[video_feat_path], ggs, gas, gis,None)
        

        bgg += bleu_g
        bga += bleu_a
        bgi += bleu_i

        #generated_sentence = ' '.join(generated_words)
        #print(gs, '\n')

        print(ggs)
        print(gas)
        print(gis)

    auc = metrics.roc_auc_score(pred_i, true_i)
    print(auc)
    print(bgg/len(test_videos), bga/len(test_videos), bgi/len(test_videos))


if __name__ =="__main__":

    test()
