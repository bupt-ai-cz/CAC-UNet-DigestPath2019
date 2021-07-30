from ..registry import LOSS


def build_loss(cfg, is_discriminator=False):
    assert cfg.MODEL.PREDICTOR.LOSS in LOSS, \
        "cfg.MODEL.PREDICTOR.LOSS: {} are not registered in registry".format(
            cfg.MODEL.PREDICTOR.LOSS
        )
    if is_discriminator:
        return LOSS[cfg.MODEL.DISCRIMINATOR.LOSS]
    return LOSS[cfg.MODEL.PREDICTOR.LOSS]
    
