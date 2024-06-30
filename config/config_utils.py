import logging
import sys_config_back


def print_configure(logger):
    # with open(cfg_file_name, 'r') as f:
    #     cfg_lines = f.readlines()

    cfg_str = ''
    for k, v in sys_config_back.sys_cfg:
        cfg_str.join('{0}:{1}\n'.format(k, v))

    dash_line = '-' * 60 + '\n'
    logger.info('System config info:\n' + dash_line + '\n' + cfg_str + '\n' + dash_line)


if __name__ == '__main__':
    print(11111)
