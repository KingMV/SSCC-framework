# -*- coding: utf-8 -*-
__author__ = 'Xin Wang'


def _get_trainer(args, logger):
    name = args.method.lower()
    if name == 'mt':  #
        from ccm.cores.MT import MT_Trainer as Trainer
    elif name == 'uda':
        from ccm.cores.UDA import UDA_Trainer as Trainer
    elif name == 'sl':
        from ccm.cores.SL import SL_Trainer as Trainer
    elif name == 'stt':
        from ccm.cores.STT import STT_Trainer as Trainer
    elif name == 'vat':
        from ccm.cores.VAT import VAT_Trainer as Trainer
    else:
        raise NotImplementedError

    return Trainer(args, logger)


def run_trainer(args, logger):
    trainer = _get_trainer(args, logger)
    logger.info('Start training')
    trainer.train_process()
    logger.info('Training finished')
