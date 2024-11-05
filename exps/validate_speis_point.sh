cd ../mfnlc
python exps/train/obstacle/ris/point.py --validate \
                                        --total_timesteps 1 \
                                        --validate_freq 1 \
                                        --load_model --load_model_folder z8rm56mp \
                                        --add_subgoal_reinforce_sg_num 2 \
                                        #--validate_subgoal_video 