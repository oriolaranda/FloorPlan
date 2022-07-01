# imports
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import measurements


# color definition
cmap = {1: {0: [[255, 255, 255], '#FFFFFF'], 127: [[0, 0, 0], '#000000'], 255: [[255, 0, 0], '#FF0000'],
            -1: [[192, 192, 192], '#C0C0C0']},
        2: {0: [[210, 105, 30], '#D2691E'], 1: [[220, 20, 60], '#DC143C'], 2: [[255, 215, 0], '#FFD700'],
            3: [[30, 144, 255], '#1E90FF'], 4: [[244, 164, 96], '#F4A460'], 5: [[255, 99, 71], '#FF6347'],
            6: [[240, 128, 128], '#F08080'], 7: [[255, 160, 122], '#FFA07A'], 8: [[255, 165, 0], '#FFA500'],
            9: [[0, 255, 127], '#00FF7F'], 10: [[255, 228, 181], '#FFE4B5'], 11: [[188, 143, 143], '#BC8F8F'],
            12: [[139, 69, 19], '#8B4513'], 13: [[255, 255, 255], '#FFFFFF'], 14: [[0, 0, 0], '#000000'],
            15: [[255, 0, 0], '#FF0000'], 16: [[128, 128, 128], '#808080'], 17: [[128, 0, 0], '#800000'],
            -1: [[192, 192, 192], '#C0C0C0']},
        4: {0: [[255, 255, 255], '#FFFFFF'], 255: [[119, 136, 153], '#778899'], -1: [[192, 192, 192], '#C0C0C0']}}


def print_image(pathImage, fileImage, channelPrint=0, valuesFilter=None, save_img=False, newFolder=None):
    print(fileImage)
    im = Image.open(pathImage + str(fileImage) + '.png', 'r')  ### read
    width, height = im.size
    ##ipixels = im.load()                       ### load: ipixels has size [width,height], each element is a tuple of nchannels elements
    ##print(type(ipixels[0,0]),ipixels[0,0])
    iarray = np.asarray(im, dtype=np.int16)  ### Convert to numpy array of size [width,height,nchannels]
    # print(iarray.shape)
    #
    # Get the image of the channel we want to visualize
    #
    ichannel = iarray[:, :, channelPrint - 1]  ### The first channel has value 1
    # print(ichannel.shape)
    #
    # Filter the values of the channel:
    #   If valuesFilter == None, prints all the values in the channel
    #   If valuesFilter != None, only prints these values and the rest are set to -1
    #
    if valuesFilter is not None:
        ifilter = np.array(ichannel, dtype='bool')
        for i in range(width):
            for j in range(height):
                ifilter[i, j] = False
                if ichannel[i, j] not in valuesFilter:
                    ifilter[i, j] = True
        ichannel[ifilter] = -1
    # print(ichannel[0, 0])
    #
    # Print the (maybe filtered) image of the channel
    #
    # img = np.ones((width, height, 3))
    # img = cmap[ichannel]
    def f_color(xi, channelPrint, max_ch3):
        if channelPrint != 3:
            return cmap[channelPrint][xi][0]
        else:
            if xi == -1:
                return [192, 192, 192]
            else:
                return [int(255*xi/max_ch3), int(255*xi/max_ch3), int(255*xi/max_ch3)]

    img = np.array([[f_color(xi, channelPrint, ichannel.max()) for xi in xi_col] for xi_col in ichannel])
    if channelPrint != 3:
        legends = []
        for classes in cmap[channelPrint].keys():
            l = mpatches.Patch(color=cmap[channelPrint][classes][1], label=classes)
            legends.append(l)
        plt.legend(handles=legends, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.imshow(img)
    plt.plot()

    if save_img:
        plt.savefig(newFolder + str(fileImage) + '.png')
        plt.close()
    else:
        plt.show()
        print('h')


if __name__ == '__main__':
    # pathImage = '../DeepLayout-DataDriven/Test_imgs/'
    pathImage = '../RPLAN/floorplan_subset/'
    fileImages = [f for f in os.listdir(pathImage) if (os.path.isfile(os.path.join(pathImage, f)) and '.png' in f)]
    # fileImages = ['980.png']
    # fileImage = '../Data/pngprova_clean/IMP_82_4.png'
    channelPrint = 3      ### The first channel has value 1
    valuesFilter = None   ### None: Print all the values in the channel / [...] List of values to visualize (only these ones)
    save_img = False
    newFolder = 'visual_images/'

    if save_img:
        if (newFolder is not None) and (not os.path.exists(newFolder)):
            os.mkdir(newFolder)
            print("Directory ", newFolder, " Created ")
        else:
            print("Directory ", newFolder, " already exists or is None")

    for fileImage in fileImages:    # fileImages[1:]:
        print_image(pathImage, fileImage.split('.')[0], channelPrint, valuesFilter, save_img, newFolder)

