import json
import pickle as pkl

def check_object(o1, o2):
    if o1['synsets']!=o2['synsets']:
        print (o1['synsets'],  o2['synsets'])
        raise Exception('Synset from attribute.json and object.json not matching')
    if o1['object_id']!=o2['object_id']:
        raise Exception('Object Id from attribute.json and object.json not matching')
    if o1['names']!=o2['names']:
        raise Exception('Names from  attribute.json and object.json not matching')
    return

def get_bbox(bbox):
    bbox = bbox.split(' ')
    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = x1 + float(bbox[2])
    y2 = y1 + float(bbox[3])
    return x1, y1, x2, y2

def check_overlapping(bbox1, bbox2):
    if ((bbox1[0] > bbox2[2]) or (bbox2[0] > bbox1[2])) or ((bbox1[1] > bbox2[3]) or (bbox2[1] > bbox2[3])):
        return False
    else:
        return True

def get_iou(bbox_str1, bbox_str_list):
    bbox1 = get_bbox(bbox_str1)
    d = {}
    for b in bbox_str_list:
        bbox2 = get_bbox(b)
        if not check_overlapping(bbox1, bbox2):
             continue
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])
        interArea = (xB - xA)*(yB - yA)
        box1Area = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1])
        box2Area = (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1])
        iou = interArea / float(box1Area + box2Area - interArea)
        d[b] = iou
    return d    

visual_genome_data_dir = '/dccstor/cssblr/amrita/VisualGenome/data/raw'    
image_ids_with_coco_id = {x['image_id']:x['coco_id'] for x in json.load(open(visual_genome_data_dir+'/image_data.json')) if x['coco_id']}
nsvqa_pipeline = '/dccstor/cssblr/amrita/NSVQA_Pipeline/preprocessed_data/coco'
year = '2017'
split = 'train'
nsvqa_pipeline_dir = nsvqa_pipeline+'/'+year+'/'+split
coco_id_image_file_map = {int(x['image_id'].split('.')[0].split('_')[-1]):{'image_filename':x['image_id'], 'height':x['height'], 'width':x['width']} for x in json.load(open(nsvqa_pipeline_dir+'/image_data.json'))}
year = '2017'
split = 'val'
nsvqa_pipeline_dir = nsvqa_pipeline+'/'+year+'/'+split
coco_id_image_file_map = {int(x['image_id'].split('.')[0].split('_')[-1]):{'image_filename':x['image_id'], 'height':x['height'], 'width':x['width']} for x in json.load(open(nsvqa_pipeline_dir+'/image_data.json'))}
year = '2014'
split = 'train'
nsvqa_pipeline_dir = nsvqa_pipeline+'/'+year+'/'+split
coco_id_image_file_map.update({int(x['image_id'].split('.')[0].split('_')[-1]):{'image_filename':x['image_id'], 'height':x['height'], 'width':x['width']} for x in json.load(open(nsvqa_pipeline_dir+'/image_data.json'))})
year = '2014'
split = 'val'
nsvqa_pipeline_dir = nsvqa_pipeline+'/'+year+'/'+split
coco_id_image_file_map.update({int(x['image_id'].split('.')[0].split('_')[-1]):{'image_filename':x['image_id'], 'height':x['height'], 'width':x['width']} for x in json.load(open(nsvqa_pipeline_dir+'/image_data.json'))})
objects_data_file = visual_genome_data_dir + '/objects.json'
attributes_data_file = visual_genome_data_dir + '/attributes.json'
relationships_data_file = visual_genome_data_dir + '/relationships.json'
scene_parsed_data = {}
for d in json.load(open(objects_data_file)):
    image_id = d['image_id']
    if image_id not in image_ids_with_coco_id:
        continue
    coco_id = image_ids_with_coco_id[image_id]
    if coco_id not in coco_id_image_file_map:
        raise Exception('Image id not found in coco_id_image_file_map')
    if image_id not in scene_parsed_data:
        scene_parsed_data[image_id] = {}
    scene_parsed_data[image_id]['image_details'] = coco_id_image_file_map[coco_id]
    scene_parsed_data[image_id]['image_details']['coco_id'] = coco_id
    objects = {}
    for object in d['objects']:
        bbox_str = ' '.join([str(x) for x in [object['x'], object['y'], object['w'], object['h']]])
        if bbox_str not in objects:
            objects[bbox_str] = {'x':object['x'], 'y':object['y'], 'w':object['w'], 'h':object['h'], 'synsets':{}}
        objects[bbox_str]['synsets'][object['object_id']] = {'object_id':object['object_id'], 'synsets':object['synsets'], 'names':object['names']}
    scene_parsed_data[image_id]['object_descriptors'] = objects

for d in json.load(open(attributes_data_file)):
    image_id = d['image_id']
    if image_id not in image_ids_with_coco_id:
        continue
    for attribute in d['attributes']:
        bbox_str = ' '.join([str(x) for x in [attribute['x'], attribute['y'], attribute['w'], attribute['h']]])
        if bbox_str not in scene_parsed_data[image_id]['object_descriptors']:
            scene_parsed_data[image_id]['object_descriptors'][bbox_str]={'synsets':{attribute['object_id']:{}}}
            scene_parsed_data[image_id]['object_descriptors'][bbox_str]['synsets'][attribute['object_id']] = attribute
        object_id = attribute['object_id']
        if 'attributes' in attribute:
            scene_parsed_data[image_id]['object_descriptors'][bbox_str]['synsets'][object_id]['attributes'] = attribute['attributes']
        
for d in json.load(open(relationships_data_file)):
    image_id = d['image_id']
    if image_id not in image_ids_with_coco_id:
        continue
    for relationship in d['relationships']:
        sub = relationship['subject']
        obj = relationship['object']
        sub_bbox_str = ' '.join([str(x) for x in [sub['x'], sub['y'], sub['w'], sub['h']]])
        obj_bbox_str = ' '.join([str(x) for x in [obj['x'], obj['y'], obj['w'], obj['h']]])
        if sub_bbox_str not in scene_parsed_data[image_id]['object_descriptors']:
            scene_parsed_data[image_id]['object_descriptors'][bbox_str]={'synsets':{sub['object_id']:{}}}
            scene_parsed_data[image_id]['object_descriptors'][bbox_str]['synsets'][sub['object_id']] = sub
        if obj_bbox_str not in scene_parsed_data[image_id]['object_descriptors']:
            scene_parsed_data[image_id]['object_descriptors'][bbox_str]={'synsets':{obj['object_id']:{}}}
            scene_parsed_data[image_id]['object_descriptors'][bbox_str]['synsets'][obj['object_id']] = obj
        sub_id = sub['object_id']
        obj_id = obj['object_id'] 
        bbox_str = sub_bbox_str+'\t'+obj_bbox_str
        rel_id = str(sub_id)+'\t'+str(obj_id)
        if 'relationship_descriptors' not in scene_parsed_data[image_id]:
             scene_parsed_data[image_id]['relationship_descriptors'] = {}
        if bbox_str not in scene_parsed_data[image_id]['relationship_descriptors']:
             scene_parsed_data[image_id]['relationship_descriptors'][bbox_str] = {'synset_pairs' : {}} 
        if id not in scene_parsed_data[image_id]['relationship_descriptors'][bbox_str]['synset_pairs']:
             scene_parsed_data[image_id]['relationship_descriptors'][bbox_str]['synset_pairs'][id] = []
        scene_parsed_data[image_id]['relationship_descriptors'][bbox_str]['synset_pairs'][id].append(relationship)
        
pkl.dump(scene_parsed_data, open(nsvqa_pipeline_dir+'/scene_parsed_data_visualgenome_gold.pkl','wb'))
