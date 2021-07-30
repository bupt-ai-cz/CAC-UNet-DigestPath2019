from ..registry import DECODER

def build_decoder(cfg, input_channels):
    assert cfg.MODEL.DECODER.TYPE in DECODER, \
        "cfg.MODEL.DECODER.TYPE: {} are not registered in registry".format(
            cfg.MODEL.DECODER.TYPE
        )
    return DECODER[cfg.MODEL.DECODER.TYPE](input_channels)