import cv2
import os
import numpy as np
import pandas as pd
import skimage
from vgg16 import *
import pdb

def preprocess_frame(image, target_height=224, target_width=224):
#    pdb.set_trace()
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width*float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1]-cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height*float(target_width)/width)))
        cropping_length = int((resized_image.shape[0]-target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0]-cropping_length,:]
    return cv2.resize(resized_image, (target_height, target_width))


def main():
    num_frames = 80
    video_path = './data/UTE'
    video_save_path = './rgb_feats'
    #videos = os.listdir(video_path)
    #videos = filter(lambda x: x.endswith('.mp4'), videos)
    
    vid_list = os.listdir(video_path)
    vid_list = [os.path.join(video_path, x) for x in vid_list]
    vid_list = filter(os.path.isdir, vid_list)

    vid_info = []

    for vid in vid_list:
        _, vid_num = os.path.split(vid)
        type_list = os.listdir(vid)
        type_list = [os.path.join(vid, x) for x in type_list]
        type_list = filter(os.path.isdir, type_list)
        type_list = [x for x in type_list]
        for type in type_list:
            _, type_val = os.path.split(type)
            img_list = os.listdir(type)
            img_list = [os.path.join(type, x) for x in img_list]
            print(img_list)
            #img_list = filter(os.path.isdir, img_list)

            for img in img_list:
                path, name = os.path.split(img)
                frame_list = os.listdir(img)
                frame_list = [os.path.join(img,x) for x in frame_list]
                vid_name=vid_num+'-{:06d}'.format(int(name))+'_'+type_val
                vid_info.append({'vid': vid_name   ,'frame':frame_list})



    input_data = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="X")
    feats = CNN(input_data)
    value = feats.fc7()

    init = tf.global_variables_initializer()

    #pdb.set_trace()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.variable_scope("vgg16") as scope:
            for idx, video in enumerate(vid_info):
                vid_name = video['vid']
                frames = video['frame']


                frame_count = 0
                frame_list = []

                for frame in frames:
                    frame = cv2.imread(frame)
                    frame_list.append(frame)
                    frame_count += 1
                frame_list = np.array(frame_list)

                if frame_count >= num_frames:
                    frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
                    frame_list = frame_list[frame_indices]

                if frame_count==0:
                    print(idx, vid_name, "PASS!!")
                    pass
                else:
                    cropped_frame_list = np.array(list(map(lambda x: preprocess_frame(x), frame_list)))



                    if idx==0:
                        sess.run(init)
                        feats.load_weights('vgg16_weights.npz', sess)

                    print(idx, vid_name, cropped_frame_list.shape)
                    Save = sess.run(value, feed_dict={input_data:cropped_frame_list})

                    save_full_path = os.path.join(video_save_path, vid_name + '.npy')
                    np.save(save_full_path, Save)

if __name__=="__main__":
    main()
