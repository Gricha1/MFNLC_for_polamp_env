cd ../mfnlc
python exps/train/obstacle/ris/point.py --validate \
                                        --total_timesteps 1 \
                                        --validate_freq 1 \
                                        --load_model --load_model_folder idhvhftt \
                                        --add_subgoal_reinforce_sg_num 0 \
                                        --validate_video_idx 4 \
                                        #--validate_subgoal_video 