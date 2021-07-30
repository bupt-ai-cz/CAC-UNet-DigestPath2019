from ..registry import PREDICTOR

def build_predictor(cfg, in_channels):
    assert cfg.MODEL.PREDICTOR.TYPE in PREDICTOR, \
        "cfg.MODEL.PREDICTOR.TYPE: {} are not registered in registry".format(
            cfg.MODEL.PREDICTOR.TYPE
        )
    assert isinstance(cfg.MODEL.PREDICTOR.NUM_CLASSES, int), \
        "cfg.MODEL.PREDICTOR.NUM_CLASSES: {} are not int".format(
            cfg.MODEL.PREDICTOR.NUM_CLASSES
        )
    num_classes = cfg.MODEL.PREDICTOR.NUM_CLASSES
    predictor = PREDICTOR[cfg.MODEL.PREDICTOR.TYPE](in_channels, num_classes)
    return predictor