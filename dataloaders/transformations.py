import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(applied_types = None, New_size=(256,256)):
	## albumentations
	## can be used as basic traning and test set, without any augmentation
	if applied_types == None:
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    ToTensorV2()
		    ])

	elif applied_types == "train":
	

		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			albumentations.RandomResizedCrop(height = New_size[0], width = New_size[1], scale=(0.9, 1.1), ratio=(0.9, 1.1), p=0.25),
			albumentations.ShiftScaleRotate(shift_limit=0.0625,
										scale_limit=0.05,
										rotate_limit=15,
										p=0.25),
			albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
			albumentations.RandomGamma(gamma_limit=(60,140),p=0.25),
			ToTensorV2()])


            
	elif applied_types == "val" or applied_types == "test":

		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			ToTensorV2()
			])

	return data_transforms

def get_transform_strong_Weak(applied_types = None, New_size = (256,256)):
    
	data_transforms_w = albumentations.Compose([
		albumentations.Resize(New_size[0], New_size[1]),
		albumentations.RandomResizedCrop(height = New_size[0], width = New_size[1], scale=(0.9, 1.1), ratio=(0.9, 1.1), p=0.25),
		albumentations.ShiftScaleRotate(shift_limit=0.0625,
									scale_limit=0.05,
									rotate_limit=15,
									p=0.25),
		ToTensorV2()])
 
	data_transforms_s = albumentations.Compose([
		albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
		albumentations.RandomGamma(gamma_limit=(60,140),p=0.5),
		ToTensorV2()])
 
	return data_transforms_w, data_transforms_s

