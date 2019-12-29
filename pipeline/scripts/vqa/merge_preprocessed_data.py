from options import get_options, get_option_str
import os
import pickle as pkl

opt = get_options()
preprocessed_dir = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))
input_preprocessed_data = opt.preprocessed_annotation_file.replace('.pkl','_')
full_preprocessed_data = {}
for file in os.listdir(preprocessed_dir):
    if file.startswith(input_preprocessed_data):
        data = pkl.load(open(os.path.join(preprocessed_dir, file), 'rb'))
        common_images = set(data).intersection(set(full_preprocessed_data))
        if len(common_images)>0:
            for image in common_images:
                 full_preprocessed_data[image].extend(data[image])
                 del data[image]
            print ('Number of repeating images ', len(common_images))
        full_preprocessed_data.update(data)
        print ('Merged file :', file)
pkl.dump(full_preprocessed_data, open(os.path.join(preprocessed_dir, opt.preprocessed_annotation_file), 'wb'))

