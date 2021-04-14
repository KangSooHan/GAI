from utils import *
import tensorflow as tf
import pandas as pd
import numpy as np

def data_preprocess(df, wordtoix, n_g, n_a, n_i, types):
    G = df["Global"].values
    G = list(map(lambda x:'<gbos> ' + x, G))
    A = df["Action"].values
    A = list(map(lambda x: '<abos> ' + x, A))
    I = df["Interaction"].values
    I = list(map(lambda x: '<ibos> ' + x, I))

    captions=[]
    for idx, (each_act, each_glo, each_int) in enumerate(zip(A, G, I)):
        word_act = each_act.lower().split(' ')
        word_glo = each_glo.lower().split(' ')
        word_int = each_int.lower().split(' ')
        if len(word_glo) < n_g:
            G[idx] = G[idx] + ' <geos> '
        else:
            new_word = ''
            for i in range(n_g-1):
                new_word = new_word + word_glo[i] + ' '
            G[idx] = new_word + ' <geos> '

        if len(word_act) < n_a:
            A[idx] = A[idx] + ' <aeos> '
        else:
            new_word = ''
            for i in range(n_a-1):
                new_word = new_word + word_act[i] + ' '
            A[idx] = new_word + ' <aeos> '

        if len(word_int) < n_i:
            I[idx] = I[idx] + ' <ieos>'
        else:
            new_word = ''
            for i in range(n_i-1):
                new_word = new_word + word_int[i] + ' '
            I[idx] = new_word + ' <ieos>'

        cap = '<start> '
        if 'G' in types:
            cap = cap + G[idx]

        if 'A' in types:
            cap = cap + A[idx]

        if 'I' in types:
            cap = cap + I[idx]

        captions.append(cap)

    df['captions'] = captions

    caption_ind = []
    for cap in captions:
        word_ind = []
        for word in cap.lower().split(' '):
            if word in wordtoix:
                word_ind.append(wordtoix[word])
            else:
                word_ind.append(wordtoix['<unk>'])
        caption_ind.append(word_ind)

    df['captions_ind'] = caption_ind

    return df

def preProBuildWordVocab(datas, types, word_count_threshold=5):
    train_Global, train_Action, train_Interaction = datas['Global'].values,  datas['Action'].values, datas['Interaction'].values

    captions_list = []
    if 'G' in types:
        captions_list.extend(list(train_Global))
    if 'A' in types:
        captions_list.extend(list(train_Action))
    if 'I' in types:
        captions_list.extend(list(train_Interaction))
    captions = np.array(captions_list, dtype=np.object)

    print('preprocessing word counts and creating vocab based on word count threshold %d' % word_count_threshold)
    word_counts = {}
    nsents = 0
    for sent in captions:
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

    np.save("wordtoix"+types, wordtoix)
    np.save("ixtoword"+types, ixtoword)
    np.save("bias_init_vector"+types, bias_init_vector)

    return wordtoix, ixtoword, bias_init_vector


def get_video_data(video_data_path, video_feat_path, train_ratio=0.8, valid_ratio=0.1):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data['video_path'] = video_data.apply(lambda row: row['Video']+'-{:06d}'.format(int(row['Vid_num'])) + '_' + row['Type'] +'.npy', axis=1)
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
    valid_len = int(len(unique_filenames)*(train_ratio+valid_ratio))

    train_vids = unique_filenames[:train_len]
    valid_vids = unique_filenames[train_len:valid_len]
    test_vids = unique_filenames[valid_len:]

    train_data = new_data[new_data['video_path'].map(lambda x: x in train_vids)]
    valid_data = new_data[new_data['video_path'].map(lambda x: x in valid_vids)]
    test_data = new_data[new_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, valid_data, test_data


def generator_fn(vid_path, cap, cap_ind):
    for vp, captions, y in zip(vid_path, cap, cap_ind):
        decoder_input, y = y[:-1], y[1:]

        x =  np.load(vp)
        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen), (decoder_input, y, y_seqlen, captions)


def input_fn(datas, wordtoix, batch_size, shuffle=False, len=80):
    shapes = (([None, 4096], ()),
              ([None], [None], (), ()))
    types = ((tf.float32, tf.int32),
             (tf.int32, tf.int32, tf.int32, tf.string))
    paddings = ((0.0, 0),
                (0, 0, 0, ''))

    vid_path = datas['video_path'].values.tolist()

    #vid = list(map(lambda x: np.load(x), vid_path))
    vid_path = datas['video_path'].values.tolist()
    cap = datas['captions'].values.tolist()
    cap_ind = datas['captions_ind'].values.tolist()

    cap_ind = tf.keras.preprocessing.sequence.pad_sequences(cap_ind, padding='post', maxlen=len)
    #vid = np.vstack(vid)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(vid_path, cap, cap_ind))

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(caption_path, video_path, n_v, n_g, n_a, n_i, batch_size, types, shuffle=False, dataset="train"):
    train_datas, valid_datas, _ = get_video_data(caption_path, video_path)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(train_datas,types, word_count_threshold=0)
    train_datas = data_preprocess(train_datas, wordtoix, n_g, n_a, n_i, types)
    valid_datas = data_preprocess(valid_datas, wordtoix, n_g, n_a, n_i, types)

    if dataset=="train":
        batches = input_fn(train_datas, wordtoix, batch_size, shuffle=shuffle, len=n_g+n_a+n_i)
        num_batches = calc_num_batches(len(train_datas), batch_size)
    elif dataset=="valid":
        batches = input_fn(valid_datas, wordtoix, batch_size, shuffle=shuffle, len=n_g+n_a+n_i)
        num_batches = calc_num_batches(len(valid_datas), batch_size)

    return batches, num_batches, len(train_datas), wordtoix, ixtoword 
