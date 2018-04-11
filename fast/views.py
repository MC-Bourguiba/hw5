from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import skfmm
import pyfmm
import time
from scipy.interpolate import interp1d
import requests
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image, io
# Create your views here.
from django.template import loader
from PIL import Image, ImageDraw
import numpy as np

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
def index(request):
    date = request.GET['date']
    n = int(request.GET['n'])
    input_json = requests.get('https://ce290-hw5-weather-report.appspot.com/?date=2018-03-03').json()
    input_x = input_json['centroid_x']
    input_y = input_json['centroid_y']
    input_r = input_json['radius']

    n_list = []
    dist_list = []
    t_list = []
    # Create constants: (i = number 'pixels' per axis for greater resolution and will be reflected in dx and number of grid spaces
    # throughout code by changing the constant multiplied here, storm_ converts the value of inputs into the scale provided by n
    # and i, time_start is used to get computational complexity)
    time_start = time.clock()

    i = 10 * n
    storm_x = input_x * (i / n)
    storm_y = input_y * (i / n)
    storm_r = input_r * (i / n)

    # Create meshgrid
    Y, X = np.meshgrid(np.linspace(0, n, i + 1), np.linspace(0, n, i + 1))
    phi_o = np.ones((i + 1, i + 1))
    phi_o[0][0] = 0

    # Circule mask created and applied to phi_o
    y, x = np.ogrid[0:i + 1, 0:i + 1]
    mask = (x - storm_x) ** 2 + (y - storm_y) ** 2 <= storm_r ** 2
    phi_masked_o = np.ma.MaskedArray(phi_o, mask)

    # Calculate the fast marching distance using phi_masked_o
    skfmm_dist_o = skfmm.distance(phi_masked_o, dx=float(n / i))

    # Commented out plot lines here and below to accurately measure computational complexity of path-finding
    # plt.contour(skfmm_dist_o, 20)
    # plt.axis('equal')
    # plt.show()

    # The fast marching approach is done again, except with the "origin" placed at the true destination. This will return the
    # distance between the destination and all other points in the grid
    phi_d = np.ones((i + 1, i + 1))
    phi_d[i][i] = 0

    # Apply same mask from earlier onto new phi_d
    phi_masked_d = np.ma.MaskedArray(phi_d, mask)

    # Calculate the fast marching distance using phi_masked_d
    skfmm_dist_d = skfmm.distance(phi_masked_d, dx=float(n / i))

    # plt.contour(skfmm_dist_d, 20)
    # plt.axis('equal')
    # plt.show()

    skfmm_dist_t = skfmm_dist_o + skfmm_dist_d

    path = np.zeros((2, (2 * i)))
    dist_travelled = 0
    k = i
    j = i
    steps_available = (2 * i) - 1
    while (k != 0) & (j != 0):
        val_side = skfmm_dist_t[k - 1][j]
        val_below = skfmm_dist_t[k][j - 1]
        if (val_side < val_below):
            k = k - 1
            path[0][steps_available] = k
            path[1][steps_available] = j

            # dist_new is the additional distance from previous point to current poinnt, so we get the distance from the
            # destination (skfmm_dist_d[k][j]) for our new spot and subtract the total distance travelled. Then, total
            # distance travelled adds on the new portion. This is done similarly in the else statement below:

            dist_new = skfmm_dist_d[k][j] - dist_travelled
            dist_travelled = dist_travelled + dist_new
        else:
            j = j - 1
            path[0][steps_available] = k
            path[1][steps_available] = j

            dist_new = skfmm_dist_d[k][j] - dist_travelled
            dist_travelled = dist_travelled + dist_new

        steps_available = steps_available - 1

    time_stop = time.clock()
    time_tot = time_stop - time_start

    # Plot the shortest path onto the distance matrix total (combined origin and destination distances)
    path_x = path[1, :]
    path_y = path[0, :]
    fig, ax = plt.subplots()
    cax = ax.imshow(skfmm_dist_t, cmap='viridis', origin='lower')
    ax.set_title('Combined Fast Marching Distance Propagation')
    cbar = fig.colorbar(cax)
    plt.plot(path_x, path_y, color='red')
    im = fig2img(fig)
    draw = ImageDraw.Draw(im)  # create a drawing object that is
    # used to draw on the new image
    red = (0, 0, 0)  # color of our text
    text_pos = (10, 10)  # top-left position of our text
    text = "date : "+str(date)+" n : "+str(n)  # text to draw
    # Now, we'll do the drawing:
    draw.text(text_pos, text, fill=red)

    del draw  # I'm done drawing so I don't need this anymore

    # We need an HttpResponse object with the correct mimetype
    response = HttpResponse(content_type="image/png")
    # now, we tell the image to save as a PNG to the
    # provided file-like object
    im.save(response, 'PNG')

    return response  # and we're done!
