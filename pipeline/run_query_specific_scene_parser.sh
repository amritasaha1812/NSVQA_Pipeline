jbsub -cores 1+0 -err err/e$1.txt -out out/o$1.txt -q x86_1h -mem 200g python -m scene_parsing.query_specific_scene_parser --num_splits 100 --split_number $1
