import numpy as np
import edt
from scipy.ndimage import median_filter
from skimage.transform import resize,rescale
from skimage import measure, segmentation,morphology,filters, feature
from skimage.exposure import adjust_gamma as ag
from skimage import measure
from ..utils import linalg

__all__ = ['roi2multilabel','roi2multimasks','prediction2seed','prediction2foreground','np_method',
           'Patchifier','normalize_image','dist2labels_simp']

def roi2multilabel(roi_file,image_shape,dst,erosion_radius=2,subdivide_degree = 2):
    import read_roi as rr
    import tifffile as tf
    from skimage import draw,morphology
    from skimage.measure import subdivide_polygon
    roi = rr.read_roi_zip(roi_file)
    canvas = np.zeros(image_shape)
    core = np.zeros(image_shape)
    for k,v in roi.items():
        xy = np.array([v['y'],v['x']]).T
        xy = subdivide_polygon(xy, degree=subdivide_degree, preserve_ends=True)
        mask = draw.polygon2mask(image_shape,xy)
        eroded = morphology.binary_erosion(mask,morphology.disk(erosion_radius))
        core[eroded==1]=1
    dilated = morphology.binary_dilation(core,morphology.disk(erosion_radius+1))
    canvas[(dilated-core)>0]=125
    canvas[core>0]=254
    tf.imsave(dst,canvas.astype(np.uint8),imagej=True)
    
def roi2multimasks(roi_file,image_file,dst,erosion_radius=2, subdivide_degree = 2):
    import read_roi as rr
    import tifffile as tf
    from skimage import draw,morphology,measure
    from skimage.segmentation import watershed
    from skimage.measure import subdivide_polygon
    roi = rr.read_roi_zip(roi_file)
    image = tf.imread(image_file)
    image_shape = image.shape
    canvas = np.zeros(image_shape)
    core = np.zeros(image_shape)
    for k,v in roi.items():
        xy = np.array([v['y'],v['x']]).T
        xy = subdivide_polygon(xy, degree=subdivide_degree, preserve_ends=True)
        mask = draw.polygon2mask(image_shape,xy)
        eroded = morphology.binary_erosion(mask,morphology.disk(erosion_radius))
        core[eroded==1]=1
        canvas[mask==1]=1
    labeled = watershed(image,markers=measure.label(core),
                        mask=canvas,watershed_line=False,compactness=100)
    tf.imsave(dst,labeled.astype(np.uint16),imagej=True)
    
    
def prediction2seed(pred_multilabel_mask,seed_min=0.3,edge_max=0.75,min_seed_size=20,opening_radius=1):
    seed=(pred_multilabel_mask[:,:,2]>seed_min)*(pred_multilabel_mask[:,:,1]<edge_max)*1
    if isinstance(opening_radius,int) and opening_radius>0:
        seed=morphology.binary_opening(seed,morphology.disk(opening_radius))
    seed=morphology.remove_small_objects(seed,min_seed_size)
    return measure.label(seed)


def dist2labels_simp(dist,mask,
                     dist_threshold=0.25,
                     mask_threshold=0.8,
                     opening=2,
                     min_particle_size=10,
                     watershedline=False):
    from skimage import measure,segmentation,morphology
    basin = 1-dist
    seed = measure.label(dist>dist_threshold)
    seed = morphology.remove_small_objects(seed,min_size=min_particle_size)
    binary_mask = mask>mask_threshold
    if int(opening)>0:
        binary_mask = morphology.binary_opening(binary_mask,morphology.disk(int(opening)))
    watersheded = segmentation.watershed(basin,
                                         markers=seed,
                                         mask=binary_mask,
                                         compactness=100,
                                         watershed_line=watershedline)
    return watersheded


def prediction2foreground(pred_multilabel_mask,channel=0,
                          threshold=0.4,
                          erosion_radius=1):
    fg=(pred_multilabel_mask[:,:,channel]<threshold)*1
    #fg=morphology.binary_closing(fg,morphology.disk(1))
    if isinstance(erosion_radius,int) and erosion_radius>0:
        fg=morphology.binary_erosion(fg,morphology.disk(erosion_radius))
    return fg*1


def distance(m1,m2):
    return np.sqrt(np.sum(np.square(m1[:,np.newaxis,:]-m2[np.newaxis,:,:]),axis=-1))


def compute_dist(multi_mask,smooth_factor=10,max_v=7):
    from skimage import measure
    contours = [] 
    canvas = np.zeros(multi_mask.shape)
    
    for i in np.unique(multi_mask):
        if i>0:
            u_mask = (multi_mask==i)*1
            xy = np.array(np.where(u_mask>0)).T
            c=measure.find_contours(median_filter(multi_mask==i,3),level=0.5)
            if len(c)==1:
                smoothed = linalg.spline_approximation(c[0],smooth_factor=smooth_factor,n=len(c[0]))
                dis = np.min(distance(xy,smoothed),axis=1)
            else:
                dis = edt.edt(u_mask)[xy[:,0],xy[:,1]]
            canvas[xy[:,0],xy[:,1]]=dis
    return normalize_img(canvas,min_perc=0,max_perc=100,min_v=0,max_v=7)


def np_method(data,method='mean',**kwargs):
    if method in ['default','mean','average','MEAN','Mean','Average']:
        return np.mean(data,**kwargs)
    elif method in ['median','Median','MEDIAN']:
        return np.median(data,**kwargs)
    elif method in ['Max','max','MAX']:
        return np.max(data,**kwargs)
    elif method in ['Min','min','MIN']:
        return np.min(data,**kwargs)
    else:
        print('method not found, use np.mean instead')
        return np.mean(data,**kwargs)
              
def image2predict(img,model,size=256,channels=3,pad=16,batch_size=5):
    shape = img.shape[:2]
    patchifier = Patchifier(shape,size,pad)
    patches = patchifier.pachify(img)
    pred = model.predict(patches, batch_size=batch_size)
    stitched = patchifier.unpatchify(pred,channels)
    return stitched

def pad_nonzero(img,size):
    h = max(img.shape[0],size)
    w = max(img.shape[1],size)
    canvas=np.zeros((h,w))
    canvas[:img.shape[0],:img.shape[1]] = img
    return canvas

def normalize_image(img,
                    mask=None,
                    min_perc=0.5,
                    max_perc=99.5,
                    min_v = 0,
                    max_v = 30000,
                    bg=0.5):
    
    """ 
    inherited from @kevinjohncutler with modifications
    perform either percentile based normalization or 
    fixed range normalization or 
    masked normalizatoin.
    
    For masked normalization:
    @ kevinjohncutler
    Normalize image by rescaling from 0 to 1 and then adjusting gamma to bring 
    average background to specified value (0.5 by default).
    
    :params img: input two-dimensional image
    :params mask: input labels or foreground mask or anything but not none
    :params min_perc: lower bound of the percentile norm (0-100)
    :params max_perc: higher bound of the percentile norm (0-100)
    :params min_v: lower bound of the absolute value for normalization, overwrite min_perc,
    :params max_v: higher bound of the absolute value for normalization, overwrite max_perc,
    :params bg: background value in the range 0-1
    :return: gamma-normalized array with a minimum of 0 and maximum of 1
    """
    if mask is None:
        th1, th2 = np.percentile(img,min_perc),np.percentile(img,max_perc)
        if min_v is not None:
            th1 = min_v
        if max_v is not None:
            th2 = max_v
        img = (img-th1)/(th2-th1)
        img[img>1] = 1
        img[img<0] = 0
        return img
    
    else:
        img = (img-img.min())/(img.max()-img.min())
        try:
            img = img**(np.log(bg)/np.log(np.mean(img[morphology.binary_erosion(mask==0)])))
        except:
            # just in case when mask is invalid
            mask = img < filters.threshold_isodata(img)
            img = img**(np.log(bg)/np.log(np.mean(img[morphology.binary_erosion(mask==0)])))
        
    return img


class Patchifier:
    """
    A simple way to convert 2D images to patches and stitch them back into one
    Currently it only works with images with shapes like (height, width) or (height, width, channel), it doesn't work
    on image series such as (frame, height, width, channel)
    The smoothing function for overlap edges is simply the mean values of the overlapping pixels. Future updates may
    consider implementing 2D spline interpolation based smoothing method, such as:
    https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
    """

    def __init__(self, img_shape=(512, 512), patch_size=128, pad=32):

        """
        :param img_shape: the original shape of the large input image
        :param patch_size: the height and width of the square-shaped patch
        :param pad: half-width of the overlapping region of neighboring patches
        """

        self._shape = img_shape
        self.size = patch_size
        self.pad = pad
        self.shape = (max(self._shape[0], self.size),
                      max(self._shape[1], self.size))
        self.pad_h = max(0, self.size - self._shape[0])
        self.pad_w = max(0, self.size - self._shape[1])
        self.ref_coords = self.generate_patch_coords()

    def generate_patch_coords(self):
        """
        funtion to generate patch coords
        :return:
        """
        h, w = self.shape
        xs = list(np.arange(0, h - self.size, self.size - 2 * self.pad)) + [h - self.size]
        if len(xs) > 1:
            if xs[-1] == xs[-2]:
                xs = xs[:-1]
        ys = list(np.arange(0, w - self.size, self.size - 2 * self.pad)) + [w - self.size]
        if len(ys) > 1:
            if ys[-1] == ys[-2]:
                ys = ys[:-1]
        ref_coords = np.array([[x, y, np.random.randint(2)] for x in xs for y in ys])
        return ref_coords

    def pachify(self, img, random_rotate=False):
        """
        convert img to patches
        :param img: src image
        :param random_rotate: if randomly rotate clips, this shouldn't be used for prediction but can be helpful for training
        :return: clipped patches
        """
        if self.shape != img.shape[:2]:
            self.__init__(img.shape[:2])
        pad_config = np.zeros((len(img.shape), 2))
        pad_config[0][1] = self.pad_h
        pad_config[1][1] = self.pad_w
        pad_config = pad_config.astype(int)
        if self.pad_h > 0 or self.pad_w > 0:
            padded_img = np.pad(img.copy(), pad_config, mode='constant')
        else:
            padded_img = img.copy()
        patches = []
        for x, y, t in self.ref_coords:
            p = padded_img[x:x + self.size, y:y + self.size]
            if random_rotate and t:
                p = p.T
            patches.append(p)
        return np.array(patches)

    def unpatchify(self, patches, n_channel):
        """
        stitch patches back into one
        :param patches: array of patches
        :param n_channel: number of channels, for instance, for a rgb image n_channel should be 3
        :return:
        """
        canvas = np.zeros(list(self.shape) + [n_channel])
        canvas_counter = np.zeros(self.shape)
        for i, p in enumerate(patches):
            x, y = self.ref_coords[i][0], self.ref_coords[i][1]
            canvas[x:x + self.size, y:y + self.size] += p
            canvas_counter[x:x + self.size, y:y + self.size] += 1
        mean_canvas = canvas / canvas_counter[:, :, np.newaxis]
        return mean_canvas[:self._shape[0], :self._shape[1]]