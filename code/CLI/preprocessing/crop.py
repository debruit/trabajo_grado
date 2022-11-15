## =========================================================================
## @author Juan Sebastián Ruiz Bulla (ruizju@javeriana.edu.co)
## @author David Alejandro Castillo Chíquiza (castillo_da@javeriana.edu.co)
## @author Oscar David Falla Pulido (falla_o@javeriana.edu.co)
## =========================================================================

import numpy as np

def crop_img(img):
  img = img[85:245, :, :]
  if (img.shape[1] != 96 or img.shape[2] != 128):
      temp_sz_y = img.shape[1] - 96
      szs_y = int(temp_sz_y/2)
      img = img[:, szs_y+25:(img.shape[1] - szs_y) + 25, 70:198]
      
  img = np.stack((img,), axis=-1)

  return img