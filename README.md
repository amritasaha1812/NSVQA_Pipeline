
1. Git clone https://github.ibm.com/cognitive-machine-vision/NSVQA_Pipeline/
2. Git clone https://github.ibm.com/cognitive-machine-vision/Concept_Catalog_VQA
3. Git clone git@github.ibm.com:cognitive-machine-vision/Dynamic-Concept-Learner.git
4. Put them in the same folder 
5. cd NSVQA_Pipeline/pipeline/
6. For gold bbox, gold object, attribute detection (from VisualGenome), make sure in the options.py
	Option ‘bbox_detection_type’  is set to ‘gold’
	Option ‘object_detection_type’ is set to ‘gold’
	Option ‘attribute_detection_type’ is set to ‘gold’
	Option ‘vqa_dataset’ is set to ‘vg_intersection_coco’

7. Run python -m scene_parsing.query_specific_scene_parser   (Skip this step if you want to reuse the existing preprocessed data)
8. Run python -m scripts.vqa.dump_program_dataset (This is required for generating the vocab file in the next step. If you do not need to re-create the vocab file, ignore this step)
9. Run python -m scripts.vqa.preprocess_questions --program_annotation_file program_annotation.pkl --output_h5_file train_questions.h5 --output_vocab_json vocab.json  (Skip this step if you want to reuse the .h5 and vocab files in the existing preprocessed folder)
      (when rerunning again with the same vocabulary use:  python -m scripts.vqa.preprocess_questions --program_annotation_file program_annotation.pkl --output_h5_file train_questions.h5 —input_vocab_json vocab.json) 
10. Change line 2 of Dynamic-Concept-Learner/program_induction/executors/__init__.py to 'from .vqa.executor import Executor' in order to use vqa specific executor 
11. Run python -m scripts.vqa.dummy_exec (Run this step to get the executor's output on the existing preprocessed form of the scene-parsed data)

12. Download preprocessed_data.zip from https://drive.google.com/file/d/1Yy4fG-KiPOAVe-rZ9i51_e7eTCnX4Jrn/view?usp=sharing and extract it here
