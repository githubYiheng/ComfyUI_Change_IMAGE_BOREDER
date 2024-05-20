import numpy as np
from PIL import Image
import torch
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
class ChangeImageBorder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "value": ("INT", {"default": 0}),
                "border_size": ("INT", {"default": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "change_border_to"
    CATEGORY = "Tools"

    def change_border_to(self, image, value, border_size):
        tensors = []
        if len(image) > 1:
            print('aaaaa')
            for img in image:

                pil_image = None
                # PIL Image
                pil_image = tensor2pil(img)
                
                pil_image = pil_image.convert("RGBA")
                # Get the image data
                data = np.array(pil_image)
        
                # Change the top and bottom border pixels
                data[:border_size, :] = [value, value, value, 255]  # top
                data[-border_size:, :] = [value, value, value, 255]  # bottom
        
                # Change the left and right border pixels
                data[:, :border_size] = [value, value, value, 255]  # left
                data[:, -border_size:] = [value, value, value, 255]  # right
        
                # Create a new image from the modified data
                new_img = Image.fromarray(data, mode='RGBA')
                # Output image
                out_image = (pil2tensor(new_img) if pil_image else img)

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:
            print('bbbb')
            pil_image = None
            img = image
            # PIL Image
            pil_image = tensor2pil(img)
            pil_image = pil_image.convert("RGBA")
            # Get the image data
            data = np.array(pil_image)
        
            # Change the top and bottom border pixels
            data[:border_size, :] = [value, value, value, 255]  # top
            data[-border_size:, :] = [value, value, value, 255]  # bottom
        
            # Change the left and right border pixels
            data[:, :border_size] = [value, value, value, 255]  # left
            data[:, -border_size:] = [value, value, value, 255]  # right
        
            # Create a new image from the modified data
            new_img = Image.fromarray(data, mode='RGBA')
            # Output image
            out_image = (pil2tensor(new_img) if pil_image else img)

            tensors = out_image

        return (tensors, )
        

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ChangeImageBorder": ChangeImageBorder
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChangeImageBorder": "Change Image Border"
}
