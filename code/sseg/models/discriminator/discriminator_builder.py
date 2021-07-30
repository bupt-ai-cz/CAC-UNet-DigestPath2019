from ..registry import DISCRIMINATOR

def build_discriminator(cfg, encoder_channels, decoder_channels, predictor_channels):
    channels_dict={}
    for x in cfg.MODEL.DISCRIMINATOR.TYPE:
        if "Encoder" in x:
            channels_dict[x] = encoder_channels
        elif "Decoder" in x:
            channels_dict[x] = decoder_channels
        elif "Predictor" or "Semantic"in x:
            channels_dict[x] = predictor_channels
            
    return {x: DISCRIMINATOR[x](
        channels=channels_dict[x],
        ) for x in cfg.MODEL.DISCRIMINATOR.TYPE}