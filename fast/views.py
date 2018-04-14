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
    import time
    start_time = time.time()
    input_json = requests.get('https://ce290-hw5-weather-report.appspot.com/?date='+str(date)).json()
    input_x = input_json['centroid_x']
    input_y = input_json['centroid_y']
    input_r = input_json['radius']

    # Create constants: (s = scaled value of spaces, based on the length input (we use 20 per instructions) and n --> can be applied
    # throughout code by changing the constant multiplied here, storm_ converts the value of inputs into the scale provided by n
    # and s, time_start is used to get computational complexity)

    s = 20 * n
    storm_x = input_x * n
    storm_y = input_y * n
    storm_r = input_r * n

    # Create phi, generated from the origin (following logic of example in https://pythonhosted.org/scikit-fmm/)
    phi_o = np.ones((s + 1, s + 1))
    phi_o[0][0] = 0

    # Open grid created of same size as phi, then circular mask created and applied to phi_o
    y, x = np.ogrid[0:s + 1, 0:s + 1]
    mask = (x - storm_x) ** 2 + (y - storm_y) ** 2 <= storm_r ** 2
    phi_masked_o = np.ma.MaskedArray(phi_o, mask)

    # Calculate the fast marching distance using phi_masked_o
    skfmm_dist_o = skfmm.distance(phi_masked_o)

    # The fast marching approach is done again, except with the "origin" placed at the true destination. This will return the
    # distance between the destination and all other points in the grid
    phi_d = np.ones((s + 1, s + 1))
    phi_d[s][s] = 0

    # Apply same mask from earlier onto new phi_d
    phi_masked_d = np.ma.MaskedArray(phi_d, mask)

    # Calculate the fast marching distance using phi_masked_d
    skfmm_dist_d = skfmm.distance(phi_masked_d)

    skfmm_dist_t = skfmm_dist_o + skfmm_dist_d
    path = np.zeros((2, (2 * s)))

    # Start counters k and j off at the constant s. Instead of changing s, iterate down on k and j until we've reached the origin
    # Steps available represents the total number of steps horizontal/vertical
    k = s
    j = s
    steps_available = (2 * s) - 1

    while (k != 0) & (j != 0):
        val_side = skfmm_dist_t[k - 1][j]
        val_below = skfmm_dist_t[k][j - 1]
        if (val_side < val_below):
            k = k - 1
            path[0][steps_available] = k
            path[1][steps_available] = j
        else:
            j = j - 1
            path[0][steps_available] = k
            path[1][steps_available] = j
        steps_available = steps_available - 1


    # Plot the shortest path onto the distance matrix total (combined origin and destination distances)
    path_x = path[1, :]
    path_y = path[0, :]
    end_time = time.time()
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
    text = "date : "+str(date)+" n : "+str(n) + " response time : " + str(np.round((end_time-start_time)/60,3)) + " min"  # text to draw
    # Now, we'll do the drawing:
    draw.text(text_pos, text, fill=red)

    del draw  # I'm done drawing so I don't need this anymore

    # We need an HttpResponse object with the correct mimetype
    response = HttpResponse(content_type="image/png")
    # now, we tell the image to save as a PNG to the
    # provided file-like object
    im.save(response, 'PNG')

    return response  # and we're done!
