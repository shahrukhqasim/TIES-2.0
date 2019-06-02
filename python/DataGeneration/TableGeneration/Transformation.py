
from skimage import io,img_as_ubyte
from skimage import transform
import numpy as np
from PIL import Image,ImageDraw
import scipy as misc
import os


def find_new_points(matrix,x,y):
    pts=np.array([[x,y,0]])
    ret=np.dot(matrix,pts.T)
    return round(ret[0][0]),round(ret[1][0])

def resize_image(img,offsets,size):
    a4im = Image.new('RGBA',
                     (int(size[0]), int(size[1])),  # A4 at 72dpi
                     (255, 255, 255,255))  # White
    a4im.paste(img, offsets)  # Not centered, top-left corner
    return a4im

def pad_original_image(img,transformation_matrix,max_width,max_height):
    points=[[0,0],[0,max_height],[max_width,0],[max_width,max_height]]
    minx=0
    miny=0
    maxx=max_width
    maxy=max_height
    for pt in points:
        x,y=find_new_points(transformation_matrix,pt[0],pt[1])
        if(x<minx):
            minx=x
        if(y<miny):
            miny=y
        if(x>maxx):
            maxx=x
        if(y>maxy):
            maxy=y

    new_width=maxx-minx
    new_height=maxy-miny
    retx = 0
    rety = 0

    offsetx=0
    offsety=0
    if (minx < 0):
        retx = minx
        offsetx=int(abs(minx))
    if (maxx > max_width):
        retx = maxx - max_width
        offsetx=0

    if (miny < 0):
        rety = miny
        offsety=int(abs(miny))
    if (maxy > max_height):
        rety = maxy - max_height
        offsety=0

    a4im=resize_image(img,(offsetx,offsety),(new_width,new_height))

    return a4im,offsetx,offsety

import random
import string

def Transform(img,bboxes,shearval,rotval,max_width,max_height):
    bboxes=np.array(np.array(bboxes))
    othersinfo=bboxes[:,:2]
    bboxes=np.array(np.array(bboxes)[:,2:],dtype=np.int64)

    afine_tf = transform.AffineTransform(shear=shearval,rotation=rotval)
    points_transformation = transform.AffineTransform(shear=-1*shearval,rotation=-1*rotval)
    #img,offsetx,offsety=pad_original_image(Image.fromarray(img.astype(np.uint8)),points_transformation.params,max_width,max_height)
    img, offsetx, offsety = pad_original_image(img, points_transformation.params,
                                               max_width, max_height)

    min_pts=bboxes[:,:2]
    max_pts=bboxes[:,2:]

    offsets = np.tile(np.array([float(offsetx), float(offsety)]), (len(min_pts),1))
    min_pts=min_pts+offsets
    max_pts=max_pts+offsets

    ones=np.tile(np.array([1.0]),(len(min_pts),1))

    min_pts=np.concatenate((min_pts,ones),axis=1)
    max_pts=np.concatenate((max_pts,ones),axis=1)

    min_pts=np.dot(min_pts,points_transformation.params.T)[:,:2]
    max_pts = np.dot(max_pts, points_transformation.params.T)[:,:2]

    transformed_bboxes=np.concatenate((min_pts,max_pts),axis=1)

    transformed_image = transform.warp(img, inverse_map=afine_tf)
    out=img_as_ubyte(transformed_image)
    out=Image.fromarray(out)
    width,height=out.size
    new_width = max_width
    new_height = new_width * height / width
    out.thumbnail((new_width,new_height),Image.ANTIALIAS)

    transformed_bboxes=np.array(transformed_bboxes)
    transformed_bboxes[:,0]=(transformed_bboxes[:,0]/width)*new_width
    transformed_bboxes[:,1]=(transformed_bboxes[:,1]/height)*new_height
    transformed_bboxes[:,2] = (transformed_bboxes[:,2] / width) * new_width
    transformed_bboxes[:,3] = (transformed_bboxes[:,3] / height) * new_height


    outbbox=out.getbbox()
    out=resize_image(out,(outbbox[0],outbbox[1]),size=(max_width,max_height))
    transformed_bboxes=np.array(transformed_bboxes,dtype=np.int64)
    transformed_bboxes=np.concatenate((othersinfo,transformed_bboxes),axis=1)
    return out,transformed_bboxes


    # draw = ImageDraw.Draw(out)
    #
    # for bbox in transformed_bboxes:
    #     draw.rectangle(((bbox[0],bbox[1]), (bbox[2],bbox[3])),outline=(0,0,255))



