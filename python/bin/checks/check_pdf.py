import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import cv2


def make_image(data, outputname, size=(10, 5.62)):
    fig = plt.figure(figsize=size, dpi=300)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    plt.imshow(data)

    pdf = matplotlib.backends.backend_pdf.PdfPages(outputname)
    for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()


image = cv2.imread('/Users/shahrukhqasim/Documents/sanity_boxes.png')


make_image(image, ('/Users/shahrukhqasim/Documents/sanity_boxes.pdf'))

