import logging
logger = logging.getLogger('base')


def create_model(opt):
    # from .model import DDPM as M
    import model.model as M
    m = getattr(M, opt['model'].get('name', 'DDPM'))(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
