cd ../mfnlc
python exps/train/obstacle/ris/point.py --validate \
                                        --validate_subgoal_video \
                                        --total_timesteps 1 \
                                        --validate_freq 1 \
                                        --load_model --load_model_folder qs56nwgh