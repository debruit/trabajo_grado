def crop_img(img):
  img = img[85:245, :, :]
  if (img.shape[1] != 96 or img.shape[2] != 128):
      temp_sz_y = img.shape[1] - 96
      szs_y = int(temp_sz_y/2)
      img = img[:, szs_y+25:(img.shape[1] - szs_y) + 25, 70:198]

  return img