import PIL.Image as Image
import cv2
import math
import numpy as np

class VQAUtils():

   def __init__(self, image_dir, split, year):
        self.image_dir = image_dir
        self.split = split
        self.year = year
        self.desired_size = 256

   def coco_image_id_to_filename(self, image_id):
        image_file = self.image_dir+'/COCO_'+self.split+self.year+'_'+str(image_id).rjust(12, '0')+'.jpg'
        return image_file

   def get_image_region(self, image_id, bbox):
        image_file = self.coco_image_id_to_filename(image_id)
        image = cv2.imread(image_file)
        x, y, w, h = bbox
        bbox_image = image[y:y+h, x:x+w]
        try:
            cropped_image = Image.fromarray(bbox_image)
        except:
            cropped_image = Image.fromarray(image)
        old_size = cropped_image.size
        ratio = float(self.desired_size)/max(old_size)
        new_size = tuple([int(math.ceil(x*ratio)) for x in old_size])
        cropped_image = cropped_image.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (self.desired_size, self.desired_size))
        new_im.paste(cropped_image, ((self.desired_size-new_size[0])//2,
                            (self.desired_size-new_size[1])//2))
        new_im = np.transpose(np.asarray(new_im, dtype=np.float32), (2,0,1))
        return new_im
