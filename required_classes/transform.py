import albumentations.augmentations as AA
import albumentations.pytorch as Ap
from albumentations.core.composition import Compose, OneOf

meanFour = (0.5, 0.5, 0.5, 0.5)

def transform_js(mode,resol):
    if mode == 'train':
        train_transform = Compose([
            AA.Resize(height = resol, width = resol),
             
            AA.Normalize(meanFour, meanFour, max_pixel_value=1.0), #normalize 방식을 바꿔볼 수도 있겠음!
            #  OneOf([
            #               AA.HorizontalFlip(p=1),
            #               AA.VerticalFlip(p=1)], p=0.5),            
            Ap.transforms.ToTensorV2()
             ])        
        return train_transform
    
    elif mode == 'test':
        test_transform = Compose([
            AA.Resize(height = resol, width = resol),
            AA.Normalize(meanFour, meanFour, max_pixel_value=1.0),
            Ap.transforms.ToTensorV2()
             ])
        return test_transform