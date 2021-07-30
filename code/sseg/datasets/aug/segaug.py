from albumentations import (
    Blur,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomBrightness,    
    RandomGamma,
    GaussianBlur,
    RandomCrop,
    RandomScale,
    PadIfNeeded,
    RGBShift,
    RandomResizedCrop
)


def seg_aug(image, mask):
    aug = Compose([
              HorizontalFlip(p=0.5),
              RandomRotate90(p=0.5),              
              RandomBrightnessContrast(p=0.2, brightness_limit=0.05, contrast_limit=0.05),
              GridDistortion(distort_limit=0.1,p=0.3),
              ])

    augmented = aug(image=image, mask=mask)
    return augmented