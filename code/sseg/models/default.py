from yacs.config import CfgNode as CN
 
_C = CN()

# save model chechkpoint and traning log to work_dir
_C.WORK_DIR = "./"

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# "Generalized_Segmentor", "UDA_Segmentor"
_C.MODEL.TYPE = 'Generalized_Segmentor'

_C.MODEL.BACKBONE = CN()
# backbone: ResNet or EfficientNet
_C.MODEL.BACKBONE.TYPE = "R-50-C1-C5"
_C.MODEL.BACKBONE.PRETRAINED = True
# IBN: only ResNet
_C.MODEL.BACKBONE.WITH_IBN = False

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.TYPE = "UnetDecoder-D5"

_C.MODEL.PREDICTOR = CN()
_C.MODEL.PREDICTOR.TYPE = "BasePredictor"
_C.MODEL.PREDICTOR.LOSS = "BCE"
_C.MODEL.PREDICTOR.NUM_CLASSES = 2

_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.LOSS = "BCEWithLogits"
_C.MODEL.DISCRIMINATOR.TYPE = []
_C.MODEL.DISCRIMINATOR.WEIGHT = []
_C.MODEL.DISCRIMINATOR.SMOOTH = False
_C.MODEL.DISCRIMINATOR.LR = []
_C.MODEL.DISCRIMINATOR.UPDATE_T = 1.0



# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.ITER_VAL = 1
_C.TRAIN.EPOCHES = 100
_C.TRAIN.OPTIMIZER = "SGD"
_C.TRAIN.BATCHSIZE = 2
_C.TRAIN.ITER_REPORT = 50
_C.TRAIN.LR = 0.001

_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.APEX_OPT = 'O1'
_C.TRAIN.EARLY_STOPPING = -1
_C.TRAIN.SAVE_ALL = False

_C.TRAIN.WEIGHT = []
_C.TRAIN.POS_WEIGHT = []

_C.TRAIN.SCHEDULER = ""
_C.TRAIN.MULTISTEPLR = CN()
_C.TRAIN.MULTISTEPLR.GAMMA = 1.0
_C.TRAIN.MULTISTEPLR.MILESTONES = []

_C.TRAIN.COSINEANNEALINGLR = CN()
_C.TRAIN.COSINEANNEALINGLR.T_MAX = 1
_C.TRAIN.COSINEANNEALINGLR.T_MULT = 1.0


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = ''
_C.DATASET.ANNS = ''
_C.DATASET.IMAGEDIR = ''
_C.DATASET.WITH_EMPTY = True
_C.DATASET.RESIZE_SCALE = 1.0
_C.DATASET.RESIZE_SIZE = []
_C.DATASET.CENTER_CROP = 1.0
_C.DATASET.USE_AUG = False
_C.DATASET.NORMALIZE = False
_C.DATASET.NUM_SAMPLES = -1

_C.DATASET.VAL = CN()
_C.DATASET.VAL.TYPE = ''
_C.DATASET.VAL.ANNS = ''
_C.DATASET.VAL.IMAGEDIR = ''
_C.DATASET.VAL.CENTER_CROP = 1.0
_C.DATASET.VAL.RESIZE_SCALE = 1.0
_C.DATASET.VAL.RESIZE_SIZE = [] # WH
_C.DATASET.VAL.REDUCE = 'sum'

_C.DATASET.TARGET = CN()
_C.DATASET.TARGET.TYPE = ''
_C.DATASET.TARGET.ANNS = ''
_C.DATASET.TARGET.IMAGEDIR = ''



cfg = _C

# test
if __name__ == "__main__":
    cfg.merge_from_file("config/test.yaml")
    cfg.freeze()
 
    cfg2 = cfg.clone()
    cfg2.defrost()
    cfg2.MODEL.PREDICTOR.NUM_CLASSES = 8
    cfg2.freeze()
 
    print("cfg:")
    print(cfg)
    print("cfg2:")
    print(cfg2)
