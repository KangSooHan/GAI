import tensorflow as tf
import numpy as np
from modules import label_smoothing, get_token_embeddings

from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp, word2idx, idx2word):
        self.hp = hp
        self.token2idx, self.idx2token = word2idx, idx2word
        self.d_model = hp.d_model
        self.embeddings = get_token_embeddings(896, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens = xs

            # src_masks

            # embedding

            enc= tf.layers.dense(x, self.d_model)
            #src_masks = tf.math.equal(mask, 0) # (N, T1)
            src_masks = tf.sequence_mask(seqlens)

            #enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            #enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, 80)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, src_masks

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, 80)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention", )

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention", )
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, src_masks = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=896))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.learning_rate, global_step, 4000)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<gbos>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(80)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        return y_hat



class S2S():
    def __init__(self, hp, wordtoix, ixtoword):
        self.hp = hp
        self.embeddings = get_token_embeddings(len(wordtoix), self.hp.d_hidden, zero_pad=True)


    def encode(self, xs, training=True):




        self.n_words = n_words
        self.n_caption = hp.n_global + hp.n_action + hp.n_interaction

#        self.w_enc_out =  tf.Variable(tf.random_uniform([dim_hidden, dim_hidden]), dtype=tf.float32, name='w_enc_out')
#        self.w_dec_state =  tf.Variable(tf.random_uniform([dim_hidden, dim_hidden]), dtype=tf.float32, name='w_dec_state')
#        self.v = tf.Variable(tf.random_uniform([dim_hidden, 1]), dtype=tf.float32, name='v')
#
        self.Wemb = tf.Variable(tf.random_uniform([n_words, hp.d_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(hp.d_hidden, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(hp.d_hidden, state_is_tuple=True)

        self.lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=hp.dropout)
        self.lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=hp.dropout)

        self.encode_image_W = tf.Variable(tf.random_uniform([hp.d_image, hp.d_hidden], -0.1,0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([hp.d_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([hp.d_hidden, n_words], -0.1,0.1), name='embed_word_W')

        self.video = tf.placeholder(tf.float32, [None, hp.n_video, hp.d_image])
        self.video_mask = tf.placeholder(tf.int32, [None])
        self.caption = tf.placeholder(tf.int32, [None, self.n_caption+1])
        self.caption_mask = tf.placeholder(tf.float32, [None, self.n_caption+1])

    def build_model(self):
        video_flat = tf.reshape(self.video, [-1, self.hp.d_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [-1, self.hp.n_video, self.hp.d_hidden])

        self.video_pad = tf.zeros([self.hp.batch_size, self.hp.n_video, self.hp.d_hidden])
        self.cap_pad = tf.zeros([self.hp.batch_size, self.n_caption, self.hp.d_hidden])

        loss = 0.0
        ###### encoding
        with tf.variable_scope("encoding_vid", reuse=tf.AUTO_REUSE) as scope:
            output1, state1 = tf.nn.dynamic_rnn(self.lstm1, image_emb,sequence_length=self.video_mask, dtype=tf.float32)

        with tf.variable_scope("encoding_cap", reuse=tf.AUTO_REUSE) as scope:
            output2, state2 = tf.nn.dynamic_rnn(self.lstm2, tf.concat([self.video_pad, output1],2), sequence_length=self.video_mask, dtype=tf.float32)


        with tf.variable_scope("decoding_vid", reuse=tf.AUTO_REUSE) as scope:
            output1, state1 = tf.nn.dynamic_rnn(self.lstm1, self.cap_pad, dtype=tf.float32)

        with tf.variable_scope("decoding_cap", reuse=tf.AUTO_REUSE) as scope:
            for i in range(self.n_caption):
                current_embed = tf.nn.embedding_lookup(self.Wemb, self.caption[:,i])
                output2, state2 = self.lstm2(tf.concat([current_embed, output1[:,i,:]], 1), state2)

                labels = self.caption[:, i+1]

                logit_words = tf.matmul(output2, self.embed_word_W)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit_words)
                cross_entropy = cross_entropy * self.caption_mask[:,i+1]

                p = tf.argmax(logit_words, 1)

                if i==0:
                    prob=tf.expand_dims(p, -1)
                else:
                    prob = tf.concat([prob, tf.expand_dims(p, -1)], -1)

                current_loss = tf.reduce_sum(cross_entropy)/self.hp.batch_size
                loss = loss + current_loss
            loss /= self.n_caption

        return loss, self.video, self.video_mask, self.caption, self.caption_mask, prob


    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, src_masks = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries



