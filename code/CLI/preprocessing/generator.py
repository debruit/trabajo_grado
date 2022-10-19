import numpy as np

def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
                      
            images.append(image)
    images = np.asarray(images)
    
    return(images)




def image_gen(img_folder, imgs, mask_folder, masks, batch_size):

    L = len(img_folder)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_folder, imgs[batch_start:limit])
            Y = load_img(mask_folder, masks[batch_start:limit])

            yield (X,Y) 

            batch_start += batch_size   
            batch_end += batch_size
