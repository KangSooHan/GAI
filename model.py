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
        self.embeddings = get_token_embeddings(self.hp.word, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, video_path = xs

            # src_masks

            # embedding
            enc= tf.layers.dense(x, self.d_model)
            #src_masks = tf.math.equal(mask, 0) # (N, T1)
            src_masks = tf.sequence_mask(seqlens)

            #enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            #enc *= self.hp.d_model**0.5 # scale

            enc  /= self.hp.d_model ** 0.5

            enc += positional_encoding(enc, self.hp.n_video)

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

            dec += positional_encoding(dec, self.hp.caption)

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
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.word))
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

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<start>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.caption)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        return y_hat, xs[2]

class S2S():
    def __init__(self, hp, word2idx, idx2word):
        self.hp = hp
        self.token2idx, self.idx2token = word2idx, idx2word
        self.d_model = hp.d_model
        self.embeddings = get_token_embeddings(self.hp.word, self.hp.d_model, zero_pad=True)

        if hp.lstm_type=='bi':
            self.lstmb = tf.nn.rnn_cell.BasicLSTMCell(hp.d_model, state_is_tuple=True)

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(hp.d_model, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(hp.d_model, state_is_tuple=True)

    def encode(self, xs, training=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            if training==True:
                if self.hp.lstm_type=='bi':
                    lstmb =  tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0-self.hp.dropout_rate)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0-self.hp.dropout_rate)
                lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1.0-self.hp.dropout_rate)
            else:
                if self.hp.lstm_type=='bi':
                    lstmb =  tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0)
                lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1.0)
            x, seqlens, video_path = xs

            enc= tf.layers.dense(x, self.d_model)

            vid_pad = tf.zeros_like(enc)
            src_masks = tf.sequence_mask(seqlens)

            with tf.variable_scope("encoding_vid", reuse=tf.AUTO_REUSE) as scope:
                if self.hp.lstm_type=='bi':
                    output1, state1 = tf.nn.bidirectional_dynamic_rnn(lstm1, lstmb, enc, sequence_length=seqlens, dtype=tf.float32)
                    output1 = tf.concat(output1, 2)
                else:
                    output1, state1 = tf.nn.dynamic_rnn(lstm1, enc, sequence_length=seqlens, dtype=tf.float32)

            with tf.variable_scope("encoding_cap", reuse=tf.AUTO_REUSE) as scope:
                output2, state2 = tf.nn.dynamic_rnn(lstm2, tf.concat([vid_pad, output1],2), sequence_length=seqlens, dtype=tf.float32)

            self.state1 = state1
            self.state2 = state2

        memory = output2
        return memory, src_masks


    def decode(self, ys, memory, src_masks, training=True, step=-1):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            if training==True:
                if self.hp.lstm_type=='bi':
                    lstmb =  tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0-self.hp.dropout_rate)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0-self.hp.dropout_rate)
                lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1.0-self.hp.dropout_rate)
            else:
                if self.hp.lstm_type=='bi':
                    lstmb =  tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0)
                lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1.0)

            decoder_inputs, y, seqlens, sents2 = ys
            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)

            cap_pad = tf.zeros_like(dec)
            with tf.variable_scope("decoding_vid", reuse=tf.AUTO_REUSE) as scope:
                if self.hp.lstm_type=='bi':
                    output1, state1 = tf.nn.bidirectional_dynamic_rnn(lstm1, lstmb, cap_pad, sequence_length=seqlens,initial_state_fw = self.state1[0], initial_state_bw = self.state1[1], dtype=tf.float32)
                    output1 = tf.concat(output1, 2)
                else:
                    output1, state1 = tf.nn.dynamic_rnn(self.lstm1, cap_pad, sequence_length=seqlens,initial_state=self.state1, dtype=tf.float32)

            with tf.variable_scope("decoding_cap", reuse=tf.AUTO_REUSE) as scope:
                output2, state2 = tf.nn.dynamic_rnn(self.lstm2, tf.concat([dec,output1], 2), sequence_length=seqlens,initial_state=self.state2, dtype=tf.float32)

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', output2, weights) # (N, T2, vocab_size)
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
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.word))
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

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<start>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for step in tqdm(range(self.hp.caption)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False, step)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        return y_hat, xs[2]

class S2T():
    def __init__(self, hp, word2idx, idx2word):
        self.hp = hp
        self.token2idx, self.idx2token = word2idx, idx2word
        self.d_model = hp.d_model
        self.embeddings = get_token_embeddings(self.hp.word, self.hp.d_model, zero_pad=True)

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(hp.d_model, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(hp.d_model, state_is_tuple=True)

        if hp.lstm_type=='bi':
            self.lstmb = tf.nn.rnn_cell.BasicLSTMCell(hp.d_model, state_is_tuple=True)

    def encode(self, xs, training=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            if training==True:
                if self.hp.lstm_type=='bi':
                    lstmb =  tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0-self.hp.dropout_rate)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0-self.hp.dropout_rate)
                lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1.0-self.hp.dropout_rate)
            else:
                if self.hp.lstm_type=='bi':
                    lstmb =  tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1.0)
                lstm2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1.0)
            x, seqlens, video_path = xs

            enc= tf.layers.dense(x, self.d_model)

            vid_pad = tf.zeros_like(enc)
            src_masks = tf.sequence_mask(seqlens)

            with tf.variable_scope("encoding_vid", reuse=tf.AUTO_REUSE) as scope:
                if self.hp.lstm_type=='bi':
                    output1, state1 = tf.nn.bidirectional_dynamic_rnn(lstm1, lstmb, enc, sequence_length=seqlens, dtype=tf.float32)
                    output1 = tf.concat(output1, 2)
                else:
                    output1, state1 = tf.nn.dynamic_rnn(lstm1, enc, sequence_length=seqlens, dtype=tf.float32)

            output2, state2 = tf.nn.dynamic_rnn(lstm2, output1, sequence_length=seqlens, dtype=tf.float32)

        memory = output2
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

            dec += positional_encoding(dec, self.hp.caption)
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
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.word))
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

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<start>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.caption)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        return y_hat, xs[2]

