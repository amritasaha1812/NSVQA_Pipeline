import os
import sys
sys.path.append('../..')
from Concept_Catalog_VQA.multiclass_unilabel_cce_synset.options import get_options as get_object_options
from Concept_Catalog_VQA.multiclass_unilabel_cce_synset.model import get_model as get_object_model
from Concept_Catalog_VQA.multiclass_unilabel_cce_attribute.datasets import get_dataset as get_object_dataset
from Concept_Catalog_VQA.multiclass_unilabel_cce_attribute.datasets import get_dataloader as get_object_dataloader
from dataset.vqa_object_dataset import VQAObjectDataset
from utils.utils import softmax

class VQAObjectCatalog():
    def __init__(self, opt, processed_vqa_dataset, processed_vqa_queries, vqa_dataset):
        self.opt = opt
        self.processed_vqa_dataset = processed_vqa_dataset
        self.processed_vqa_queries = processed_vqa_queries
        self.vqa_dataset = vqa_dataset
        self.load_object_catalog()

    def load_object_catalog(self):
        self.object_catalog_model_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.object_catalog_model_dir)
        self.object_catalog_preprocessed_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.object_catalog_preprocessed_dir)
        self.vqa_object_dataset = VQAObjectDataset(None, None, None, self.processed_vqa_dataset, self.processed_vqa_queries, self.object_catalog_preprocessed_dir, None, None, None, None, self.opt.gpu_ids, 0, 0, '', self.opt)
        opt = get_object_options('test')
        opt.load_checkpoint_path = self.object_catalog_model_dir
        opt.dump_checkpoint_path = self.object_catalog_model_dir
        opt.preprocessed_data_dir = self.object_catalog_preprocessed_dir
        opt.dataset = 'vqa'
        vocab_size = self.vqa_object_dataset.vocab_size
        cluster_size = self.vqa_object_dataset.cluster_size
        clusters = self.vqa_object_dataset.clusters_np
        self.object_catalog_model = get_object_model(opt, cluster_size, vocab_size, clusters)
        print ('Loading model with cluster size ',cluster_size, ' Vocab size ', vocab_size)
        self.vqa_object_dataloader = get_object_dataloader(self.vqa_object_dataset, opt)
   
    def execute(self):
        for image_id, question_index, bbox_index, bbox, image_region, context_glove_emb, context_attention in self.vqa_object_dataloader:
            if image_region is None:
                continue
            self.object_catalog_model.set_input(context_glove_emb, context_attention, image_region)
            self.object_catalog_model.forward()
            scores = self.object_catalog_model.get_pred()
            object_distribution_scores = softmax(scores, axis=1)
            object_distribution_scores = np.concatenate((np.zeros((object_distribution_scores.shape[0],1)), object_distribution_scores), axis=1) 
            self.add_object_catalog_features_to_bbox(image_id, question_index, bbox_index, image_region, object_distribution_scores)

    def add_object_catalog_features_to_bbox(self, image_id, question_index, bbox_index, image_region, object_distribution):
        for i in range(len(image_id.tolist())):
            d = {}
            d['image_region'] = image_region[i]
            d['object_distribution'] = object_distribution[i]
            self.vqa_dataset[(int(image_id[i]), int(question_index[i]), int(bbox_index[i]))] = d
