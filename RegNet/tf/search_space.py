import os
import sys
import yaml
import copy
import random
import argparse

from model.complexity import get_complexity


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str,       default='./config.yml')
    parser.add_argument("--num-model",      type=int,       default=100)
    parser.add_argument('--baseline-path',  type=str,       default=None)
    parser.add_argument('--model-name',     type=str,       help='AnyNetXA')

    return parser.parse_args()

def select_config(cfg):
    new_cfg = copy.deepcopy(cfg)
    for k, v in new_cfg.items():
        if k == 'n_block':
            assert len(cfg[k]) == 2, 'n_block\'s length must be 2.'
            assert cfg[k][0] > 0, 'n_block\'s min value must be larger than 0.'
            new_cfg[k] = [random.choice(range(cfg[k][0], cfg[k][1]+1)) for _ in range(new_cfg['n_stage'])]

        elif k == 'n_channel':
            assert len(cfg[k]) == 2, 'n_channel\'s length must be 2.'
            assert cfg[k][0] % 8 == 0, 'n_channel\'s min value must be divisible by 8.'
            new_cfg[k] = [random.choice(range(cfg[k][0], cfg[k][1], 8)) for _ in range(new_cfg['n_stage'])]

        elif k in ['bottleneck_ratio', 'group_width']:
            new_cfg[k] = random.choice(v)

        else:
            new_cfg[k] = v

    ##########################
    # Constraint
    ##########################
    # Whole : n_channels must be divisible by 8.
    for c in new_cfg['n_channel']:
        if not (c//new_cfg['bottleneck_ratio']) % new_cfg['group_width'] == 0:
            # log = 'The number of features({}, {}) must be divisible by group_width({}).'.format(
            #     c, new_cfg['bottleneck_ratio'], new_cfg['group_width'])
            return new_cfg, False

    # TODO : Add constraints of each model
    # AnyNetXB

    # AnyNetXC

    # AnyNetXD

    # AnyNetXE


    return new_cfg, True

def check_overlap(path, whole_cfg, new_cfg):
    stamp_list = os.listdir(path)
    for k, v in whole_cfg.items():
        if v == new_cfg:
            return False
    return True

def search():
    args = get_argument()
    assert args.model_name is not None, 'model_name must be set.'

    sys.path.append(args.baseline_path)
    from common import get_logger
    logger = get_logger("MyLogger")

    cfg = yaml.full_load(open(args.config, 'r'))
    whole_cfg = {}
    cnt = 0
    os.makedirs('./result/{}/{}'.format(cfg['dataset'], args.model_name), exist_ok=True)
    while cnt < args.num_model:
        new_cfg, flag1 = select_config(cfg)
        cx = get_complexity(new_cfg)
        if flag1:
            if cfg['min_flops'] <= cx['flops'] <= cfg['max_flops']:
                logger.info('cnt : {} | cfg : {} | cx : {}'.format(cnt, new_cfg, cx))
                new_cfg['flops'] = cx['flops']
                new_cfg['params'] = cx['params']
                new_cfg['acts'] = cx['acts']
                flag2 = check_overlap('./result/{}/{}'.format(cfg['dataset'], args.model_name), whole_cfg, new_cfg)
                if flag2:
                    os.makedirs('./result/{}/{}/{}'.format(cfg['dataset'], args.model_name, cnt), exist_ok=True)
                    yaml.dump(
                        new_cfg, 
                        open('./result/{}/{}/{}/model_desc.yml'.format(cfg['dataset'], args.model_name, cnt), 'w'),
                        default_flow_style=False)
                    whole_cfg[cnt] = new_cfg
                    cnt += 1

if __name__ == '__main__':
    search()