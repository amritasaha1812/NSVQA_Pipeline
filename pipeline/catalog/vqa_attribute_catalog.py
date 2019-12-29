import os
import sys
sys.path.append('../..')
from Concept_Catalog_VQA import multiclass_unilabel_cce_synset, multiclass_unilabel_cce_attribute
from Concept_Catalog_VQA.multiclass_unilabel_cce_attribute.options import get_options as get_attribute_options
from Concept_Catalog_VQA.multiclass_unilabel_cce_attribute.model import get_model as get_attribute_model
from Concept_Catalog_VQA.multiclass_unilabel_cce_synset.datasets import get_dataset as get_attribute_dataset
from Concept_Catalog_VQA.multiclass_unilabel_cce_synset.datasets import get_dataloader as get_attribute_dataloader
from dataset.vqa_attribute_dataset import VQAAttributeDataset
from utils.utils import softmax

class VQAAttributeCatalog():
    def __init__(self, opt, processed_vqa_dataset, processed_vqa_queries, vqa_dataset):
        self.opt = opt
        self.processed_vqa_dataset = processed_vqa_dataset
        self.processed_vqa_queries = processed_vqa_queries
        self.vqa_dataset = vqa_dataset
        self.load_attribute_catalog()

    def load_attribute_catalog(self):
        self.attr_catalog_model_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.attribute_catalog_model_dir)
        self.attr_catalog_preprocessed_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.attribute_catalog_preprocessed_dir)
        self.vqa_attribute_dataset = VQAAttributeDataset(None, None, None, self.processed_vqa_dataset, self.processed_vqa_queries, self.attr_catalog_preprocessed_dir, None, None, None, None, self.opt.gpu_ids, 0, 0, '', self.opt)
        opt = get_attribute_options('test')
        opt.load_checkpoint_path = self.attr_catalog_model_dir
        opt.dump_checkpoint_path = self.attr_catalog_model_dir
        opt.preprocessed_data_dir = self.attr_catalog_preprocessed_dir
        opt.dataset = 'vqa'
        vocab_size = self.vqa_attribute_dataset.vocab_size
        cluster_size = self.vqa_attribute_dataset.cluster_size
        clusters = self.vqa_attribute_dataset.clusters_np
        self.attribute_catalog_model = get_attribute_model(opt, cluster_size, vocab_size, clusters)
        print ('Loading model with cluster size ',cluster_size, ' Vocab size ', vocab_size)
        self.vqa_attribute_dataloader = get_attribute_dataloader(self.vqa_attribute_dataset, opt)
    
    def execute(self):
        for image_id, question_index, bbox_index, bbox, image_region, context_glove_emb, context_attention in self.vqa_attribute_dataloader:
            if image_region is None:
                continue
            self.attribute_catalog_model.set_input(context_glove_emb, context_attention, image_region)
            self.attribute_catalog_model.forward()
            scores = self.attribute_catalog_model.get_pred()
            attribute_distribution_scores = softmax(scores, axis=1)
            attribute_distribution_scores = np.concatenate((np.zeros((attribute_distribution_scores.shape[0],1)), attribute_distribution_scores), axis=1)
            self.add_attribute_catalog_features_to_bbox(image_id, question_index, bbox_index, image_region, attribute_distribution_scores) 

    def add_attribute_catalog_features_to_bbox(self, image_id, question_index, bbox_index, image_region, attribute_distribution):
        for i in range(len(image_id.tolist())):
            d = {}
            d['image_region'] = image_region[i]
            d['attribute_distribution'] = attribute_distribution[i]
            self.vqa_dataset[(int(image_id[i]), int(question_index[i]), int(bbox_index[i]))] = d
