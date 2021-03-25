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
import matplotlib.pyplot as plt
import csv

import nltk

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_video, n_caption, bias_init_vector=None, attention=None, dropout_prob=0.5):
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
            score = tf.reshape(score, (self.batch_size, self.n_caption, 1))
            score = tf.nn.softmax(score, dim=-1, name='alpha')

            H_s = tf.reshape(H_s, (self.batch_size,-1, self.dim_hidden))
            C_i = tf.reduce_sum(tf.multiply(H_s, score), axis=1)
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

#                if i==0:
#                    C_i = bahdanau_attention(i)
#                else:
#                    C_i = bahdanau_attention(i, state2[1])
#
                output2, state2 = self.lstm2(tf.concat([current_embed, output1[:,i,:]], 1), state2)

                labels = self.caption[:, i+1]

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit_words)
                cross_entropy = cross_entropy * self.caption_mask[:,i+1]

                p = tf.argmax(logit_words, 1)


                if i==0:
                    prob=tf.expand_dims(p, -1)
                else:
                    prob = tf.concat([prob, tf.expand_dims(p, -1)], -1)

                current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
                loss = loss + current_loss
            loss /= self.n_caption

        return loss, self.video, self.video_mask, self.caption, self.caption_mask, prob


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


    with open('word.csv', 'w', newline='\n') as f:
        writer = csv.writer(f)
        for value in chart_values:
            writer.writerow(value)

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




def train(pretrain=False, restore_path=None, situation='G'):
    #### global parameters
    video_path = './data/UTE'
    video_feat_path = './rgb_feats'
    video_data_path = './data/UTE_caption.csv'
    model_path = './models_g'


    train_datas, _ = get_video_data(video_data_path, video_feat_path, 0.9)


    train_Global = train_datas['Global'].values
    train_Action = train_datas['Action'].values
    train_Interaction = train_datas['Interaction'].values


    captions_list = []
    if 'G' in situation:
        captions_list.extend(list(train_Global))
    if 'A' in situation:
        captions_list.extend(list(train_Action))
    if 'I' in situation:
        captions_list.extend(list(train_Interaction))
    captions = np.array(captions_list, dtype=np.object)

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)

    print(len(wordtoix))

    np.save('./data/wordtoix_g', wordtoix)
    np.save('./data/ixtoword_g', ixtoword)
    np.save('./data/bias_init_vector_g', bias_init_vector)


    #### train parameters
    dim_image = 4096
    dim_hidden = 1000
    n_video = 80
    n_gcaption = 30
    n_acaption = 20
    n_icaption = 20

    n_caption = 0
    if 'G' in situation:
        n_caption += n_gcaption
    if 'A' in situation:
        n_caption += n_acaption
    if 'I' in situation:
        n_caption += n_icaption

    n_epochs = 2000
    batch_size = 136
    learning_rate = 0.0001

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_video=n_video,
            n_caption=n_caption,)
            #bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.InteractiveSession(config=config)

    # tensorflow version 1.3
    saver = tf.train.Saver(max_to_keep=100)
#    train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(tf_loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    if pretrain:
        saver = tf.train.import_meta_graph(restore_path)
        tf.global_variables_initializer().run()
        saver.restore(sess, tf.train.latest_cehckpoint('./models'))
    else:
        tf.global_variables_initializer().run()

    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []

    test_output = open('output.txt', 'w')

    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []
        index = list(train_datas.index)
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

            current_feats = np.zeros((batch_size, n_video, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))

            current_video_masks = np.array(list(map(lambda x: len(x), current_feats_vals)), dtype=np.int32)

            for ind, feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat


            current_global = current_batch["Global"].values
            current_global = list(map(lambda x:'<gbos> ' + x, current_global))
            current_action = current_batch["Action"].values
            current_action = list(map(lambda x: '<abos> ' + x, current_action))
            current_interaction = current_batch["Interaction"].values
            current_interaction = list(map(lambda x: '<ibos> ' + x, current_interaction))

            current_captions=[]
            for idx, (each_act, each_glo, each_int) in enumerate(zip(current_action, current_global, current_interaction)):
                word_act = each_act.lower().split(' ')
                word_glo = each_glo.lower().split(' ')
                word_int = each_int.lower().split(' ')
                if len(word_glo) < n_gcaption:
                    current_global[idx] = current_global[idx] + ' <geos> '
                else:
                    new_word = ''
                    for i in range(n_gcaption-1):
                        new_word = new_word + word_glo[i] + ' '
                    current_global[idx] = new_word + ' <geos> '

                if len(word_act) < n_acaption:
                    current_action[idx] = current_action[idx] + ' <aeos> '
                else:
                    new_word = ''
                    for i in range(n_acaption-1):
                        new_word = new_word + word_act[i] + ' '
                    current_action[idx] = new_word + ' <aeos> '

                if len(word_int) < n_icaption:
                    current_interaction[idx] = current_interaction[idx] + ' <ieos>'
                else:
                    new_word = ''
                    for i in range(n_icaption-1):
                        new_word = new_word + word_int[i] + ' '
                    current_interaction[idx] = new_word + ' <ieos>'

                current_cap = '<start> '
                if 'G' in situation:
                    current_cap = current_cap + current_global[idx]

                if 'A' in situation:
                    current_cap = current_cap + current_action[idx]

                if 'I' in situation:
                    current_cap = current_cap + current_interaction[idx]

                current_captions.append(current_cap)

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

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

if __name__ =="__main__":

    train()
