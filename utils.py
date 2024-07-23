import torch

def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                label, text = line.strip().split('\t')
                labels.append(label)
                texts.append(text)
    
    return labels, texts


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda x: x.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)
    
    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)
    
    return total_norm