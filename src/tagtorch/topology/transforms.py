from demeter import euler
from demeter import directions as dirs

import numpy as np
import torch
from torch.utils.data import Dataset

'''
This class inherits from torch.Dataset. Give it a Dataset object and a (topological) transformation 
function and the resulting TAGTorchDataset will have the same items, transformed using the given 
transformation. Any attributes that the user wants to copy over from the original Dataset can be 
put in the attributes list. Any *args and **kwargs provided will be passed to the transform_fn.

Note: This function works if original_dataset is just a list where each element is a piece of data 
(e.g., an image). In this case self.metadata will remain empty. If original_dataset is a list of
tuples then it will treat original_dataset[i][0] as the data and original_dataset[i][1:] as the metadata.
'''
class TAGTorchDataset(Dataset):
    def __init__(self, original_dataset, transform_fn, limit = -1, attributes = [], *args, **kwargs):
        print(f'Transforming data using {transform_fn}...')
        self.data = []
        self.metadata = []
        
        if limit == -1:
            print('\tSetting limit to transform all data')
            limit = len(original_dataset)

        print('\tCopying attributes')
        for attr in attributes:
            if not hasattr(original_dataset, attr):
                print(f'\t  Attribute {attr} not present')
                continue
                
            if not attr.startswith('_') and not callable(getattr(original_dataset, attr)) and not attr=='data':
                # don't let the user copy attributes or functions they shouldn't copy
                 print(f'\t  Copying {attr}')
                 try:
                     setattr(self, attr, getattr(original_dataset, attr))
                 except:
                     print(f'\t   Couldn\'t copy {attr}')
                     pass  # Skip if can't copy
            else:
                print(f'\t  Shouldn\'t copy {attr}, skipping.') 
        
        print('\tComputing transformations...')
        for idx in range(len(original_dataset)):
            if idx%1000==0 and idx > 0:
                print(f'\t  Completed {idx} of {limit} transformations (full dataset has {len(original_dataset)} items)')

            if idx > limit-1:
                # only process up to a certain limit of items
                break
                
            item = original_dataset[idx]
            
            if isinstance(item, tuple):
                thing = item[0]
                meta = item[1:]
                transformed_thing = transform_fn(thing, *args, **kwargs)
                self.data.append(transformed_thing)
                self.metadata.append(meta)
            else:
                self.data.append(transform_fn(item, *args, **kwargs))
        
        print(f"Transformation complete! {len(self.data)} data items processed.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.metadata:
            return (self.data[idx],) + self.metadata[idx]
        return self.data[idx]

'''
Compute the ECT using demeter on the CubicalComplex.

Note: Now supports both NumPy and PyTorch tensors via updated demeter implementation.
'''
def demeter_ect(CComplex, num_dirs=32, num_thresh=128, device=None) -> torch.Tensor:
    """
    Compute ECT using demeter on a CubicalComplex.
    
    Args:
        CComplex: CubicalComplex instance
        num_dirs (int): Number of directions
        num_thresh (int): Number of thresholds
        device (torch.device, optional): Device for computation
    
    Returns:
        Result of ECT computation (torch.Tensor or np.ndarray depending on CComplex input)
    """
    # Use PyTorch implementation for regular_directions
    circle_dirs = dirs.regular_directions(num_dirs, dims=CComplex.ndim)
    result = CComplex.ECT(circle_dirs, T=num_thresh)
    return result

'''
Take in a grayscale image as torch 2d array, return ECT torch 3d array (with shape[0] = 1).

The `foreground` parameter indicates whether black (B) or white (W) should be considered as foreground.
The `epsilon` parameter indicates how far from pure black or pure white should be considered as foreground.
'''
def grayscale_image_to_ect(img, num_dirs = 32, num_thresh = 128, foreground = 'B', epsilon = 0.01, debug=False):
    if len(img.shape)==3 and img.shape[0] == 1:
        this_img=img[0] # or img.squeeze()?
    elif len(img.shape)==3 and img.shape[0] > 1:
        print('Image is not grayscale')
    elif len(img.shape) == 2:
        this_img=img
    else:
        print('Data item is not properly formatted as a grayscale image or is not an image')
        
    #this_img[this_img > 0] = 1 # this overwrites this_img instead of making a copy so I'm doing something different
    if foreground =='W':
        # White = 1.0 so we want to pick pixels that are within epsilon of 1.0
        new_img = torch.where(this_img > 1.0-epsilon, 1.0, 0.0)
    elif foreground == 'B':
        # Black = 0.0 so we want to pick pixels that are within epsilon of 0.0
        new_img = torch.where(this_img < 0.0+epsilon, 1.0, 0.0)
    else:
        raise ValueError('foreground flag must be either \'B\' (black) or \'W\' (white)')
    new_img = new_img.numpy() # TODO, make demeter work with torch natively? Or maybe that's not needed?

    img_complex = euler.CubicalComplex(new_img).complexify()

    ECT = demeter_ect(img_complex, num_dirs=num_dirs, num_thresh=num_thresh).reshape(num_dirs, num_thresh).T

    if isinstance(ECT, np.ndarray):
        ECT = torch.from_numpy(ECT)
    tensor = ECT.unsqueeze(0).float()

    if debug:
        return new_img, tensor
    else:
        return tensor

'''
Take in a color image as torch 3d array with 3 channels (R, G, B),
transform the image into grayscale image and then use the grayscale_image_to_ect
function to return ECT torch 3d array (with shape[0] = 1)
'''
def color_image_to_ect_via_grayscale(img, num_dirs = 32, num_thresh = 128, foreground = 'B', epsilon = 0.01, debug=False):
    # Assuming img has shape (3, H, W) transform using luminosity formula (there are multiple similar weightings, this is just one choice)
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
    gray_img = (img * weights).sum(dim=0, keepdim=True)

    if debug:
        # Use this debug flag if you want to see the transformed grayscale image, boolean image, and ect 
        bool_img, ect = grayscale_image_to_ect(gray_img, num_dirs=num_dirs, num_thresh=num_thresh, foreground=foreground, epsilon=epsilon, debug=True)
        return gray_img, bool_img, ect
    else:
        ect = grayscale_image_to_ect(gray_img, num_dirs=num_dirs, num_thresh=num_thresh, foreground=foreground, epsilon=epsilon)
        return ect

'''
Take in a color image as torch 3d array with 3 channels (R, G, B), treat each channel as its own grayscale image
transform each channel of the image into into ect using grayscale_image_to_ect
return stack of ECTs as torch 3d array (with shape[0] = 3)
'''
def color_image_to_ect_via_channels(img, num_dirs=32, num_thresh=128, foreground='B', epsilon=0.01):
    r_img = img[0]
    g_img = img[1]
    b_img = img[2]

    r_ect = grayscale_image_to_ect(r_img,  num_dirs=num_dirs, num_thresh=num_thresh, foreground=foreground, epsilon=epsilon)
    g_ect = grayscale_image_to_ect(g_img,  num_dirs=num_dirs, num_thresh=num_thresh, foreground=foreground, epsilon=epsilon)
    b_ect = grayscale_image_to_ect(b_img,  num_dirs=num_dirs, num_thresh=num_thresh, foreground=foreground, epsilon=epsilon)

    stacked = torch.cat([r_ect, g_ect, b_ect], dim=0)

    return stacked

"""
Takes in a torch.Dataset object for which the individual items are grayscale images and
returns a TAGTorchDataset object for which the individual items are ects of the images
"""
def transform_2d_grayscale_image_dataset_to_ect(data: Dataset, limit=-1, attrs=[], *args, **kwargs) -> TAGTorchDataset:
    transformed_dataset = TAGTorchDataset(data, grayscale_image_to_ect, limit, attrs, *args, **kwargs)
    return transformed_dataset

"""
Takes in a torch.Dataset object for which the individual items are color images and
returns a TAGTorchDataset object for which the individual items are ects of the images, either 
consdered as grayscale images or a stack of ects, one for each of the 3 channels of the color image
"""
def transform_3d_color_image_dataset_to_ect(data: Dataset, limit=-1, attrs=[], option='gray', *args, **kwargs) -> TAGTorchDataset:
    if option == 'gray':
        transformed_dataset = TAGTorchDataset(data, color_image_to_ect_via_grayscale, limit, attrs, *args, **kwargs)
    elif option == 'channels':
        transformed_dataset = TAGTorchDataset(data, color_image_to_ect_via_channels, limit, attrs, *args, **kwargs)
    else:
        print(f'Option \'{option}\' to treat color images is not supported')
        transformed_dataset = None

    return transformed_dataset

############### EMILIE TODO next function ################
def transform_3d_image_dataset_to_ect(data: Dataset, limit=-1, attrs=[]) -> TAGTorchDataset:
    # need to use the demeter 3D functionality that I haven't tested yet
    pass

############### Leaving as a stub for GREG TODO, PH stuff  ################
