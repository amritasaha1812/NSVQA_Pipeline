import json
import pickle as pkl
import sys
sys.path.append('/dccstor/cssblr/amrita/Mask_RCNN/')
import mrcnn
coco_path = '/dccstor/cssblr/amrita/Mask_RCNN/samples/coco'
sys.path.append(coco_path)
import coco

dataset = coco.CocoDataset()
coco_data_dir = '/dccstor/cssblr/amrita/Mask_RCNN/data/coco'
dataset.load_coco(coco_data_dir, 'train')
dataset.prepare()
coco_id_map = {}
for i, info in enumerate(dataset.class_info):
    coco_id_map[info['id']] = info['name']
    
coco_vg_name_map = {'sports ball':'ball', 'fire hydrant':'fire_extinguisher', 'stop sign':'stop',
 'handbag':'shoulder_bag', 'suitcase':'briefcase', 'sports ball':'ball', 
 'hot dog':'hotdog', 'tv':'television', 'wine glass':'wineglass', 'skis':'ski', 'donut':'doughnut',
 'couch':'sofa', 'potted plant':'houseplant', 'dining table':'dinner_table', 'remote':'remote_control',
 'cell phone':'cellular_telephone', 'teddy bear':'teddy','hair drier':'dryer', 'orange':'fruit' }

visual_genome_data_dir = '/dccstor/cssblr/amrita/VisualGenome/data/raw'
vg_objects = {}
        
for d in json.load(open(visual_genome_data_dir+'/object_types.json')):
    d1 = '.'.join(d.split('.')[:-2]).replace('"','').strip().replace(' ','_').replace('-','_')
    d2 = d.replace('_','')
    if d1 not in vg_objects:
        vg_objects[d1] = set([])
    if d2 not in vg_objects:
        vg_objects[d2] = set([])
    vg_objects[d1].add(d)
    vg_objects[d2].add(d)
  
    
vg_objects['BG'] = 'OOV'    
preprocessed_dir = '../preprocessed_data'
coco_vg_object_map = {}
coco_vg_object_id_map = {}
for k, v in coco_id_map.items():
    v_name_orig = v
    if v in coco_vg_name_map:
        v = coco_vg_name_map[v]
        print ('changing ',v_name_orig, 'to ',v)
    obj = v.replace(' ','_').replace('-','_').strip()
    obj1 = obj.replace('_','')
    vg_object_name = None
    if obj in vg_objects:
        vg_object_name  = vg_objects[obj]
    elif obj1 in vg_objects:
        vg_object_name = vg_objects[obj1]
    else:
         print ('cannot find ', v_name_orig)
    print ('Coco Object Name ', v_name_orig, ' --->', vg_object_name)
    coco_vg_object_map[v_name_orig] = vg_object_name
    coco_vg_object_id_map[k] = vg_object_name
pkl.dump(coco_vg_object_map, open(preprocessed_dir+'/coco_vg_object_map.pkl','wb'))
pkl.dump(coco_vg_object_id_map, open(preprocessed_dir+'/coco_vg_object_id_map.pkl','wb'))
