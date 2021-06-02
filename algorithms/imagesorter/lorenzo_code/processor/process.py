from __future__ import division
import os
import json
import tensorflow as tf
print(tf.__version__)
from keras.models import load_model
import argparse
from preprocess import *
import timeit


def sort_images(image_batch):
    """
    Apply Imagesorter model to the preprocessed images.

    Return:
    predictions :  list of predictions. ([type, rotation, inversion, lateral_flip])
    
    
    """
    
    predictions = []
    print('About to load model')
    imagesorter = load_model('image_sorter.hdf5')
    print('Model loaded')

    num_imgs = len(image_batch)
    for ind, img in enumerate(image_batch):
        print('about to predict for image ', ind, 'of ', num_imgs)
        
        ###if ind==3:
        ####    sitk.WriteImage(sitk.GetImageFromArray(img), '/output/tempout2.png')
        
        # print('img shape and type', img.shape, img.dtype)
        
        img = np.stack((img, img, img), axis=2)
        # print('now img shape and type', img.shape, img.dtype)
        img = img/255.
        # print('now now imge shape and type', img.shape, img.dtype)
        ####img = np.transpose(img)
        
        ###Image.fromarray(img, 'L')
        ####img = img.convert("P")
        ####if ind==3:
        ####    img.save('/output/tempout3.png')
        ####img = np.asarray(img.convert('RGB'), dtype=np.uint8)
        ####print('last shape', img.shape)
        ####if ind==3:
        ####    sitk.WriteImage(sitk.GetImageFromArray(img), '/output/tempout4.png')
        
        
        
        
        
        pred = imagesorter.predict(np.expand_dims(img,0))
        
        im_type = np.argmax(pred[0])

        im_rot = np.argmax(pred[1])

        im_inv = 1 if pred[2] > 0.5 else 0

        im_flip = 1 if pred[3] > 0.5 else 0
        
        predictions.append([im_type, im_rot, im_inv, im_flip])

    return predictions


def save_results(predictions, name_list, output_folder):

    """
    Save results in a Json file in the output folder.
    """

    type_labels = ['PA', 'AP', 'lateral', 'not-CXR']
    rotation_labels = ['0', '90', '180', '270']
    inversion_labels = ['No', 'Yes']
    lateral_flip_labels = ['No', 'Yes']

    json_data = {}



    for n, pred in enumerate(predictions):

        prediction_data = {}

        prediction_data['type'] = type_labels[pred[0]]    
        prediction_data['rotation'] = rotation_labels[pred[1]]
        prediction_data['inversion'] = inversion_labels[pred[2]]
        prediction_data['lateral_flip'] = lateral_flip_labels[pred[3]]

        json_data[name_list[n]] = prediction_data
     
    #with open(output_filename, 'w') as outfile:
    with open(os.path.join(output_folder, "results.json"), 'w') as outfile:
        json.dump(json_data, outfile, indent=4)


if __name__ == "__main__":
    print('entering main')
    start_time = timeit.default_timer()
    ap = argparse.ArgumentParser(
        prog='process.py',
        description=
            'Reads all images from an input directory and produces '
            'results in an output directory')

    ap.add_argument('input_dir', help = "input directory to process")
    ap.add_argument('output_dir', help = "output directory generate result files in")

    args = vars(ap.parse_args())

    # Check the output folder is writeable
    output_folder = args['output_dir']
    try:
        with open(os.path.join(output_folder, "tmpfile"), 'w') as outfile:
            # file opened for writing.
            print('output location is writeable')
           
    except IOError as x:
        print('error ', x.errno, ',', x.strerror)
        print('unable to write to the folder provided', output_folder)
        exit(0)

    # remove the tmp file created above
    os.remove(os.path.join(output_folder, "tmpfile"))
    
    # Get the list of valid image files (recursive) from the input folder
    print('input dir is ', args['input_dir'])
    path_list, name_list, ext_list = get_image_list(args['input_dir'])
    print('found', len(path_list), 'images to work on')
    
    #Preprocess the incoming images
    
    ready_to_test_images = preprocess_images(path_list, ext_list)
    
    print('preprocessed ', ready_to_test_images.shape[0], 'images')
    
    predictions = sort_images(ready_to_test_images)
    save_results(predictions, path_list, output_folder)
    print('done, saved results at ', output_folder)
    
    """
    print('About to load model')
    imagesorter = load_model('image_sorter.hdf5')
    print('Model loaded')
    for ind in range(0,len(path_list)):
        ready_to_test_image = preprocess_images([path_list[ind]], [ext_list[ind]])
        prediction = sort_images(ready_to_test_image, imagesorter)
        outfile = output_folder + '/' + name_list[ind] + '.json'
        save_results(prediction, [path_list[ind]], outfile)
    """   
    print('Processing took ', timeit.default_timer() - start_time)
