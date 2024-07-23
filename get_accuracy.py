import sys
import argparse

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--preds_fn', required=True)
    p.add_argument('--y_fn', required=True)

    config = p.parse_args()

    return config


def main(config):

    correct = 0

    with open(config.preds_fn, 'r') as f1, open(config.y_fn, 'r') as f2:
        preds = f1.readlines()
        y = f2.readlines()
    
    assert len(preds) == len(y), 'Prediction results and labels are incompatible.'
    for pred, y_i in zip(preds, y):
        if pred == y_i:
            correct += 1

    accuracy = correct / len(y)

    print('Test accuracy: {:.3f}'.format(accuracy))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
