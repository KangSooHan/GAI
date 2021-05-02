import argparse


class Hparams:
    parser = argparse.ArgumentParser()

    # train

    parser.add_argument('--types', default="GI", help="Global Action Interaction Types : {GAI GA GI AI G A I}")

    ## files
    parser.add_argument('--save_dir', default="t2t_GI", help="save directory")

    parser.add_argument('--log_dir', default="log", help="hparam directory")
    parser.add_argument('--eval_dir', default="eval", help="hparam directory")
    parser.add_argument('--test_dir', default="test", help="hparam directory")
    parser.add_argument('--chkpt_dir', default="model", help="model directory")
    parser.add_argument('--video_dir', default="data/UTE", help="video directory")
    parser.add_argument('--video_path', default="rgb_feats", help="features directory")
    #parser.add_argument('--caption_path', default="data/UTE_caption.csv", help="caption path")
    parser.add_argument('--caption_path', default="data/backup.csv", help="caption path")

    ##training scheme
    ### Training Param
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    parser.add_argument('--num_blocks', default=5, type=int)
    parser.add_argument('--num_heads', default=4, type=int)

    ##model
    parser.add_argument('--d_image', default=4096, type=int, help="Image embedding size")
    parser.add_argument('--d_model', default=256, type=int, help="hidden dimension of LSTM")
    parser.add_argument('--d_ff', default=1024, type=int, help="hidden dimension of LSTM")
    parser.add_argument('--n_video', default=80, type=int, help="Max length of video")
    parser.add_argument('--n_global', default=30, type=int, help="Max length of global")
    parser.add_argument('--n_action', default=20, type=int, help="Max length of action")
    parser.add_argument('--n_interaction', default=20, type=int, help="Max length of interaction")

    parser.add_argument('--lstm_type', default='bi', type=str, help="Type of LSTM")
