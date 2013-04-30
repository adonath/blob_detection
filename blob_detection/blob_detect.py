import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

from scipy.ndimage import gaussian_filter, gaussian_laplace,  maximum_filter
from scipy.ndimage.morphology import binary_erosion
from numpy import sqrt, sin, cos, pi, arccos, abs
from itertools import combinations
from scipy.misc import imread

#This basic blob detection algorithm is based on:
#http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
#Theory behind: http://en.wikipedia.org/wiki/Blob_detection (04.04.2013)
#I modified the peak detection by using a maximum filter.


def create_scale_space(image, scale_parameters, filter_ = 'gaussian_laplace'):
    """Creates Scale Space for a given image and stores it in 3D array"""
    #Filter option
    if filter_ == 'gaussian':
        scale_filter = gaussian_filter
        N = 'scale**2' #Normalisation for linear scale space
    elif filter_ == 'gaussian_laplace':
        scale_filter = gaussian_laplace
        N = '-scale**2' #Normalisation for linear scale space
    else:
        logging.error('Invalid filter option')
    
    #Set up scale parameters
    width, height = image.shape
    scale_space = np.ndarray(shape = (0, width, height))
    
    #Compute scale space
    for scale in scale_parameters:
        logging.info("Computing scale {0}:".format(scale))
        image_scaled = eval(N) * np.array(scale_filter(image, scale, mode='constant'), ndmin=3)
        scale_space = np.append(scale_space, image_scaled, axis=0)
    return scale_space        
 

def detect_peaks(image):
    """Detect peaks in an image  using a maximum filter"""
    #Set up 3x3 footprint for maximum filter
    footprint = np.ones((3, 3))

    #Apply maximum filter: All pixel in the neighborhood are set
    #to the maximal value. Peaks are where image = maximum_filter(image)
    local_maximum = maximum_filter(image, footprint=footprint) == image
    
    #We have false detections, where the image is zero, we call it background.
    #Create the mask of the background
    background = (image == 0)

    #Erode background at the borders, otherwise we would miss  
    eroded_background = binary_erosion(background, structure=footprint, border_value=1)

    #Remove the background from the local_maximum image
    detected_peaks = local_maximum - eroded_background
    return detected_peaks


def detect_peaks_3D(image):
    """Same functionality as detect_peaks, but works on image cubes"""
    #Set up 3x3 footprint for maximum filter
    footprint = np.ones((3, 3, 3))

    #Apply maximum filter: All pixel in the neighborhood are set
    #to the maximal value. Peaks are where image = maximum_filter(image)
    local_max = maximum_filter(image, footprint=footprint, mode='constant')==image
    
    #We have false detections, where the image is zero, we call it background.
    #Create the mask of the background
    background = (image==0)

    #Erode background at the borders, otherwise we would miss 
    eroded_background = binary_erosion(background, structure=footprint, border_value=1)

    #Remove the background from the local_max mask
    detected_peaks = local_max - eroded_background
    return detected_peaks
    
 
def show_peaks(image_3D):
    """Show all images of different scales including the detected peaks. Useful for debugging.""" 
    for scale_image in image_3D:
        #Detect peaks
        detected_peaks = detect_peaks(scale_image)
        
        #Show image and peaks
        plt.imshow(scale_image)
        x, y = np.where(detected_peaks== 1)
        plt.scatter(y, x)
        plt.show()
    
 
def detect_blobs_3D(image, threshold):
    """Find maxima in image cubes"""
    #Replace nan values by 0
    image = np.nan_to_num(image)
    
    #Compute scale parameters and compute scale space   
    scale_parameters = np.linspace(1, 30, 50)
    image_3D = create_scale_space(image, scale_parameters)
    blobs = []
    
    #Employ threshold
    mask_threshold = image_3D > threshold
    detected_peaks = detect_peaks_3D(image_3D * mask_threshold)
    scale_list, y_list, x_list = np.where(detected_peaks==1)
    
    #Loop over all found blobs     
    for x, y, scale in zip(x_list, y_list, scale_list):
        val = image_3D[scale][y][x]
        norm = image[y][x]
        blobs.append(Blob(x, y, scale, val, norm))
    logging.info('Found {0} blobs.'.format(len(blobs)))    
    return blobs
    
 
def detect_blobs(image, threshold):
    """Detect blobs of different sizes"""
    #Replace nan values by 0
    image = np.nan_to_num(image) 
    
    #Compute scale parameters and compute scale space
    scale_parameters = np.linspace(1, 30, 15)
    image_3D = create_scale_space(image, scale_parameters)
    blobs = []
    
    #Loop over all scale space images 
    for i, scale_image in enumerate(image_3D):
        mask_threshold = scale_image > threshold #Maybe it is useful to employ different threshold values on different scales 
        detected_peaks = detect_peaks(scale_image * mask_threshold)
        y_list, x_list = np.where(detected_peaks==1)
        for x, y in zip(x_list, y_list):
            scale = scale_parameters[i]
            val = scale_image[y][x]
            norm = image[y][x]
            blobs.append(Blob(x, y, scale, val, norm))
    logging.info('Found {0} blobs.'.format(len(blobs)))
    return blobs
        
 
def prune_blobs(blobs, overlap_threshold):
    """Prune blobs. If the overlap area of two blobs is to large, the one with the smaller peak value is dismissed"""
    #Loop over all pairwise blob combinations
    for blob_1, blob_2 in combinations(blobs, 2):
        overlap_area = blob_1.overlap(blob_2)
        overlap_percent = max(overlap_area/blob_1.area(), overlap_area/blob_2.area())
        if overlap_percent > overlap_threshold: #Overlap criterion, neighborhood criterion
                if blob_1.value >  blob_2.value: #Find maximum
                    blob_2.keep = False
                else:
                    blob_1.keep = False
    return [blob for blob in blobs if blob.keep] #That is Python programming at its best:-)


def show_blobs(image, blobs):
    """Show input image with overlaid blobs"""
    plt.imshow(image, origin='lower')
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    #plt.colorbar()
       
    for blob in blobs:
        logging.info('Found blob: {0}'.format(blob))
        x, y = blob.image() 
        plt.plot(x, y, color = 'y')
    plt.show()


class Blob(object):
    """An excess blob is represented by a position, radius and peak value."""  
    def __init__(self, x_pos, y_pos, radius, value, norm):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = 1.41 * radius #The algorithm is most sensitive for extensions of sqrt(2) * t, where t is the scale space parameter
        self.value = value
        self.norm = norm
        self.keep = True
        
        
    def area(self):
        """Compute blob area"""
        return pi * self.radius**2
    
    
    def overlap(self, blob):
        """Compute overlap area between two blobs"""
        #For now it is just the overlap area of two containment circles
        #It could be replaced by the Q or C factor, which also defines
        #a certain neighborhood. 
         
        d = sqrt((self.x_pos - blob.x_pos)**2 + (self.y_pos - blob.y_pos)**2) 
        
        #One circle lies inside the other
        if d < abs(self.radius - blob.radius):
            area = pi * min(self.radius, blob.radius)**2
        
        #Circles don't overlap    
        elif d > (self.radius + blob.radius):
            area = 0
            
        #Compute overlap area. Reference: http://mathworld.wolfram.com/Circle-CircleIntersection.html (04.04.2013)
        else:   
            area = (blob.radius**2 * arccos((d**2 + blob.radius**2 - self.radius**2)/(2 * d * blob.radius)) 
                                + self.radius**2 * arccos((d**2 + self.radius**2 - blob.radius**2)/(2 * d * self.radius)) 
                                - 0.5 * sqrt(abs((-d + self.radius + blob.radius)*(d + self.radius - blob.radius) * 
                                                 (d - self.radius + blob.radius) * (d + self.radius + blob.radius))))
        return area
    

    def image(self):
        """Return image of the blob"""
        phi = np.linspace(0, 2*pi, 360)
        x = self.radius * cos(phi) + self.x_pos
        y = self.radius * sin(phi) + self.y_pos 
        return x, y


    def __str__(self):
        """Is called by the print statement"""
        return 'x_pos: {0}, y_pos: {1}, radius: {2:02.2f}, peak value: {3:02.2f}'.format(self.x_pos, self.y_pos, self.radius, self.value)
        

if __name__=="__main__":
    #Example of the blob detection applied to the Hubble deep field
    #There are basically two parameters you can tune:
    #Threshold on the height of the peak 
    #Threshold on the overlap of neighboring peaks on different scales
    
    #Select an image section, because detecting peaks on the whole image will take too long for a short test
    x_slice = slice(0, 1000)
    y_slice = slice(750, 1250)
    
    #Load images. Blob detection works only on grey value images, but the RGB image will be shown
    image_grey = imread('HubbleDeepField.png', flatten=True)[y_slice, x_slice] 
    image_rgb = imread('HubbleDeepField.png')[y_slice, x_slice] 
    
    #Threshold parameter for Gaussian Laplace works the value 20 on the hubble deep field
    blobs = detect_blobs(image_grey, 20)
    
    #Overlap parameter for Gaussian Laplace works an overlap of 0.1 
    blobs = prune_blobs(blobs, 0.1) 
    
    #Show RGB image and with blobs overlaid
    show_blobs(image_rgb, blobs)
    
    
