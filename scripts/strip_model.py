import os
import argparse
from datetime import datetime
import torch

def predict_config(parser):
    parser.add_argument('--checkpoint', default='d:/model_4.pt')
    parser.add_argument('--fout', type=str, default='mt-dnn-cased-v2.pt')
    return parser

parser = argparse.ArgumentParser()
parser = predict_config(parser)
args = parser.parse_args()

def main():
    opt = vars(args)
    fout = args.fout
    model_path = args.checkpoint
    state_dict = None
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        config = state_dict['config']
        opt.update(config)
        if state_dict['config']['ema_opt'] > 0:
            new_state_dict = {'state': state_dict['ema'], 'config': state_dict['config']}
        else:
            new_state_dict = {'state': state_dict['state'], 'config': state_dict['config']}
        old_state_dict = {}
        for key, val in new_state_dict['state'].items():
            prefix = key.split('.')[0]
            if prefix == 'scoring_list':
                continue
            old_state_dict[key] = val
        my_config = {}
        my_config['vocab_size'] = config['vocab_size']
        my_config['hidden_size'] = config['hidden_size']
        my_config['num_hidden_layers'] = config['num_hidden_layers']
        my_config['num_attention_heads'] = config['num_attention_heads']
        my_config['hidden_act'] = config['hidden_act']
        my_config['intermediate_size'] = config['intermediate_size']
        my_config['hidden_dropout_prob'] = config['hidden_dropout_prob']
        my_config['attention_probs_dropout_prob'] = config['attention_probs_dropout_prob']
        my_config['max_position_embeddings'] = config['max_position_embeddings']
        my_config['type_vocab_size'] = config['type_vocab_size']
        my_config['initializer_range'] = config['initializer_range']
        state_dict = {'state': old_state_dict, 'config': my_config}
        torch.save(state_dict, args.fout)

if __name__ == '__main__':
    main()