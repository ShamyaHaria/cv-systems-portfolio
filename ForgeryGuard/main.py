# Shamya Haria
# CS5330 - Pattern Recognition and Computer Vision
# Entry point for training, evaluation and grad-cam

import argparse
from src.train    import train
from src.evaluate import evaluate
from src.gradcam  import visualise_pair

config = {
    'iam_root':       'data/iam',
    'words_txt':      'data/iam/words.txt',
    'num_pairs':      50000,
    'batch_size':     32,
    'epochs':         20,
    'lr':             1e-4,
    'checkpoint_dir': 'outputs/checkpoints',
    'checkpoint':     'outputs/checkpoints/best_model.pth',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate', 'gradcam'])
    parser.add_argument('--img1', default='data/iam/a01/a01-000u/a01-000u-00-00.png')
    parser.add_argument('--img2', default='data/iam/a01/a01-000u/a01-000u-00-01.png')
    parser.add_argument('--out',  default='outputs/gradcam/sample_pair.png')
    args = parser.parse_args()

    if args.mode == 'train':
        train(config)
    elif args.mode == 'evaluate':
        evaluate(config)
    elif args.mode == 'gradcam':
        visualise_pair(args.img1, args.img2, config['checkpoint'], args.out)