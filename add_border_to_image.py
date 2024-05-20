import numpy as np
from PIL import Image


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

    def change_border_to(image_path, value, border_size=2):
        # Open the image file
        with Image.open(image_path) as img:
            # Convert to RGBA if it's not already in this mode
            img = img.convert("RGBA")

            # Get the image data
            data = np.array(img)

            # Change the top and bottom border pixels
            data[:border_size, :] = [value, value, value, 255]  # top
            data[-border_size:, :] = [value, value, value, 255]  # bottom

            # Change the left and right border pixels
            data[:, :border_size] = [value, value, value, 255]  # left
            data[:, -border_size:] = [value, value, value, 255]  # right

            # Create a new image from the modified data
            new_img = Image.fromarray(data, mode='RGBA')

            # Return the new image
            return new_img

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
