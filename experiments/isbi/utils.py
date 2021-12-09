import numpy as np
import torch


def save_gif(image_tensor, path, vmin=None, vmax=None, cmap=None):
    from PIL import Image
    if torch.is_tensor(image_tensor):
        if len(image_tensor.shape) == 2:
            image_list = [image_tensor]
        else:
            image_list = torch.unbind(image_tensor)
    else:
        image_list = image_tensor
    processed = []
    for im in image_list:
        im = im.detach().cpu().numpy()
        if cmap is None:
            if vmin is None:
                vmin = im.min()
            if vmax is None:
                vmax = im.max()
            im = 255 * (im - vmin) / (vmax - vmin + 1e-6)
            im[im < 0] = 0
            im[im > 255] = 255
        else:
            assert(vmin is None)
            im = apply_segmented_cmap(im, vmax)

        im = Image.fromarray(im.astype(np.uint8))
        processed.append(im)

    if len(processed) > 1:
        processed[0].save(path + '.gif', save_all=True, append_images=processed[1:], duration=2000, loop=0, disposal=1)
    else:
        processed[0].save(path + '.png')


# Generate random colormap
def rand_cmap(nlabels, type='soft', first_color_black=True, last_color_black=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap


def apply_segmented_cmap(image_array, max_v=None):
    if torch.is_tensor(image_array):
        image_array = image_array.cpu().detach().numpy()

    if max_v is None:
        max_v = max(image_array.max(), 1)

    cm = rand_cmap(max(int(max_v), 2))
    image_array = image_array / max_v
    image_array_colored = cm(image_array)
    image_array_colored = (image_array_colored * 255).astype(np.uint8)
    return image_array_colored
