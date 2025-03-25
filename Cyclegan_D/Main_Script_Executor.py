""" From this script the entire method can be executed, the train procedure as well as the test step.
    The parameters specified here are the ones related with remote sensing application.
"""
import os

Schedule = []
#Training the Domain Adaptation
# Schedule.append("python train_remote_sensing.py --compute_ndvi False --buffer False " 
#                 "--name prove " 
#                 "--images_section Organized/Images/ --reference_section Organized/References/ "
#                 "--source_domain Amazon_PA --target_domain Cerrado_MA --stride_s 21 --stride_t 19 "
#                 "--source_image_name_T1 02_08_2016_image_R225_62 --source_image_name_T2 20_07_2017_image_R225_62 "
#                 "--target_image_name_T1 18_08_2017_image --target_image_name_T2 21_08_2018_image "
#                 "--dataroot /mnt/Data/Work/School/Trabajos_para_Tesis/Trabajo_Domain_Adaptation/Dataset/")
# # Generating the translated domains
# Schedule.append("python test_remote_sensing.py --model cycle_gan --compute_ndvi False --buffer False " 
#                 "--name prove " 
#                 "--images_section Organized/Images/ --reference_section Organized/References/ "
#                 "--source_domain Amazon_PA --target_domain Cerrado_MA --overlap_porcent_s 0.40 --overlap_porcent_t 0.40 "
#                 "--source_image_name_T1 02_08_2016_image_R225_62 --source_image_name_T2 20_07_2017_image_R225_62 "
#                 "--target_image_name_T1 18_08_2017_image --target_image_name_T2 21_08_2018_image "
#                 "--dataroot /mnt/Data/Work/School/Trabajos_para_Tesis/Trabajo_Domain_Adaptation/Dataset/")
####### original Pedro ^^^^^^^^


# mog: test run
# Schedule.append(
# 				"python mog_train.py "
# 				# "python train_remote_sensing.py "
# 				"--compute_ndvi False --buffer False " 
#                 "--name nada "                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 1  --load_size 72  --crop_size 64 "
#                 "--same_coordinates False "
#                 "--niter 50 "

#                 "--save_epoch_freq 10 "
#                 # "--use_task_loss True "
#                 # "--continue_train --epoch_count 20 "
#                 # "--num_threads 6 "

#                 "--source_domain Amazon_PA  --stride_s 21 "
#                 "--source_image_name_T1 PA/img_past_PA  --source_image_name_T2 PA/img_present_PA "
#                 "--source_reference_name_T1 PA/new_gt_past_PA  --source_reference_name_T2 PA/gt_present_PA "
#                 "--source_buffer_in 2  --source_buffer_out 0 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")


# # sRO_tMA_CN - cyclegan with all losses: adversarial, cycle and identity loss
# Schedule.append("python train_remote_sensing.py --compute_ndvi False --buffer False " 
#                 "--name prove " 
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10  --load_size 72  --crop_size 64 "
#                 "--same_coordinates False "

#                 "--checkpoints_dir './checkpoints/sRO_tMA_CN' --save_epoch_freq 10 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 "
#                 "--continue_train --epoch_count 160 "
#                 "--num_threads 6 "
                
#                 "--source_domain Amazon_RO --target_domain Cerrado_MA --stride_s 45 --stride_t 19 "
#                 "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
#                 "--target_image_name_T1 MA/img_past_MA --target_image_name_T2 MA/img_present_MA "
#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# # sRO_tMA_CT - cyclegan + diff loss
# Schedule.append("python train_remote_sensing.py --compute_ndvi False --buffer False " 
#                 "--name prove " 
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10  --load_size 72  --crop_size 64 "
#                 "--same_coordinates False "

#                 "--checkpoints_dir ./checkpoints/sRO_tMA_CD --save_epoch_freq 10 "
#                 "--lambda_diff_s 1 --lambda_diff_t 1 " # default values

#                 "--source_domain Amazon_RO --target_domain Cerrado_MA --stride_s 45 --stride_t 19 "
#                 "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
#                 "--target_image_name_T1 MA/img_past_MA --target_image_name_T2 MA/img_present_MA "
#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# mogsRO_tMA_CN - cyclegan with all losses: adversarial, cycle and identity loss
# Schedule.append(
# 				"python mog_train.py "
# 				# "python train_remote_sensing.py "
# 				"--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10  --load_size 72  --crop_size 64 "
#                 "--niter 50 "

#                 "--name mogsRO_tMA_CN "
#                 # " --save_epoch_freq 10 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 "
#                 # "--continue_train --epoch_count 10 "
#                 # "--num_threads 6 "

#                 "--source_domain Amazon_RO --stride_s 45 "
# 				"--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
# 				"--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
# 				"--source_buffer_in 2 --source_buffer_out 4 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# mog_sRO_tMA_CT - cyclegan losses + task loss
# Schedule.append(
# 				"python mog_train.py "
# 				# "python train_remote_sensing.py "
# 				"--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10  --load_size 72  --crop_size 64 "
                
#                 "--name mog_sRO_tMA_CT2 "
#                 "--niter 0 "
#                 " --save_epoch_freq 5 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 "
#                 "--use_task_loss True "
#                 # "--continue_train --epoch_count 10 "

#                 "--source_domain Amazon_RO --stride_s 45 "
# 				"--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
# 				"--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
# 				"--source_buffer_in 2 --source_buffer_out 4 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# # mog_sRO_tMA_CD - cyclegan losses + diff loss
# Schedule.append(
# 				"python mog_train.py "
# 				# "python train_remote_sensing.py "
# 				"--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10  --load_size 72  --crop_size 64 "
                
#                 "--name mog_sRO_tMA_CD "
#                 # "--niter 50 "
#                 " --save_epoch_freq 5 "
#                 "--lambda_diff_s 1 --lambda_diff_t 1 " # default
#                 # "--use_task_loss True "
#                 "--continue_train --epoch_count 151 "

#                 "--source_domain Amazon_RO --stride_s 45 "
# 				"--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
# 				"--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
# 				"--source_buffer_in 2 --source_buffer_out 4 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# # og_sRO_tMA_CD - cyclegan losses + diff loss
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10  --load_size 72  --crop_size 64 "
                
#                 "--name og_sRO_tMA_CD "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 1 --lambda_diff_t 1 " # default
#                 # "--num_threads 8 "
#                 # "--use_task_loss True "
#                	"--continue_train --epoch_count 51 "

#                 "--source_domain Amazon_RO --stride_s 45 "
#                 "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
#                 "--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
#                 "--source_buffer_in 2 --source_buffer_out 4 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")


# # # og_sRO_tMA_CT - cyclegan losses + task loss
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10 --load_size 72  --crop_size 64 "
                
#                 "--name og_sRO_tMA_CT "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 " # default
#                 "--num_threads 6 "
#                 # "--use_task_loss True "
#                 # "--continue_train --epoch_count 51 "
#                 "--print_freq 1 "

#                 "--source_domain Amazon_RO --stride_s 45 "
#                 "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
#                 "--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
#                 "--source_buffer_in 2 --source_buffer_out 4 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# # og_sRO_tMA_CDNorm - cyclegan losses + diff loss (normalizada)
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10 --load_size 72  --crop_size 64 "
                
#                 "--name og_sRO_tMA_CDNorm "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 1 --lambda_diff_t 1 " # default
#                 "--num_threads 8 "
#                 # "--use_task_loss True "
#                 "--continue_train --epoch_count 198 "
#                 # "--print_freq 1 "

#                 "--source_domain Amazon_RO --stride_s 45 "
#                 "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
#                 "--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
#                 "--source_buffer_in 2 --source_buffer_out 4 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")


# # og_sMA_tRO_CT - cyclegan losses + task loss
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10 --load_size 72  --crop_size 64 "
                
#                 "--name og_sMA_tRO_CT "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 " # default
#                 "--num_threads 7 "
#                 "--use_task_loss True "
#                 "--continue_train --epoch_count 118 "
#                 "--print_freq 1 "

#                 "--source_domain Cerrado_MA --stride_s 19 "
#                 "--source_image_name_T1 MA/img_past_MA --source_image_name_T2 MA/img_present_MA "
#                 "--source_reference_name_T1 MA/gt_past_MA --source_reference_name_T2 MA/gt_present_MA "
#                 "--source_buffer_in 0  --source_buffer_out 2  "

#                 "--target_domain Amazon_RO --stride_t 45 "
#                 "--target_image_name_T1 RO/img_past_RO --target_image_name_T2 RO/img_present_RO "
#                 "--target_reference_name_T1 RO/new_gt_past_RO --target_reference_name_T2 RO/gt_present_RO "
#                 "--target_buffer_in 2 --target_buffer_out 4 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# # # og_256_sMA_tRO_CN - patch 256; batch 1; different stride; cyclegan normal
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 1 --load_size 286  --crop_size 256 "
                
#                 "--name og_256_sMA_tRO_CN "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 " # default
#                 "--num_threads 7 "
#                 # "--use_task_loss True "
#                 # "--continue_train --epoch_count 118 "
#                 "--print_freq 1 "

#                 "--source_domain Cerrado_MA --stride_s 19 "
#                 "--source_image_name_T1 MA/img_past_MA --source_image_name_T2 MA/img_present_MA "
#                 "--source_reference_name_T1 MA/gt_past_MA --source_reference_name_T2 MA/gt_present_MA "
#                 "--source_buffer_in 0  --source_buffer_out 2  "

#                 "--target_domain Amazon_RO --stride_t 50 "
#                 "--target_image_name_T1 RO/img_past_RO --target_image_name_T2 RO/img_present_RO "
#                 "--target_reference_name_T1 RO/new_gt_past_RO --target_reference_name_T2 RO/gt_present_RO "
#                 "--target_buffer_in 2 --target_buffer_out 4 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")


# # og_sRO_tMA_CTDNorm - cyclegan + diff loss (normalizada) + task loss
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 10 --load_size 72  --crop_size 64 "
                
#                 "--name og_sRO_tMA_CTDNorm "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 1 --lambda_diff_t 1 " # default
#                 "--num_threads 0 "
#                 # "--use_task_loss True "
#                 # "--continue_train --epoch_count 76 "
#                 "--print_freq 1 "

#                 "--source_domain Amazon_RO --stride_s 45 "
#                 "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
#                 "--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
#                 "--source_buffer_in 2 --source_buffer_out 4 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")


# # og256_sPA_tMA_CN - rodada teste de cyclegan normal para patch 256
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 1 --load_size 284  --crop_size 256 "
                
#                 "--name og256_sPA_tMA_CN "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 "
#                 # "--num_threads 0 "
#                 # "--use_task_loss True "
#                 "--continue_train --epoch_count 130 "
#                 "--print_freq 50 "

#                 "--source_domain Amazon_PA  --stride_s 21 "
# 				"--source_image_name_T1 PA/img_past_PA --source_image_name_T2 PA/img_present_PA "
# 				"--source_reference_name_T1 PA/new_gt_past_PA --source_reference_name_T2 PA/gt_present_PA "
# 				"--source_buffer_in 2 --source_buffer_out 0 "

#                 "--target_domain Cerrado_MA --stride_t 19 "
#                 "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
#                 "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
#                 "--target_buffer_in 0  --target_buffer_out 2 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")

# # # og_256_sMA_tRO_CN - patch 256; batch 1; different stride; cyclegan normal
# Schedule.append(
#                 # "python mog_train.py "
#                 "python train_remote_sensing.py "
#                 "--compute_ndvi False --buffer False " 
                                
#                 "--images_section '' --reference_section '' "
#                 "--batch_size 1 --load_size 286  --crop_size 256 "
                
#                 "--name og_256_sMA_tRO_CN "
#                 # "--niter 50 "
#                 " --save_epoch_freq 1 "
#                 "--lambda_diff_s 0 --lambda_diff_t 0 " # default
#                 # "--num_threads 5 "
#                 # "--use_task_loss True "
#                 "--continue_train --epoch_count 180 "
#                 "--print_freq 100 "

#                 "--source_domain Cerrado_MA --stride_s 19 "
#                 "--source_image_name_T1 MA/img_past_MA --source_image_name_T2 MA/img_present_MA "
#                 "--source_reference_name_T1 MA/gt_past_MA --source_reference_name_T2 MA/gt_present_MA "
#                 "--source_buffer_in 0  --source_buffer_out 2  "

#                 "--target_domain Amazon_RO --stride_t 50 "
#                 "--target_image_name_T1 RO/img_past_RO --target_image_name_T2 RO/img_present_RO "
#                 "--target_reference_name_T1 RO/new_gt_past_RO --target_reference_name_T2 RO/gt_present_RO "
#                 "--target_buffer_in 2 --target_buffer_out 4 "

#                 "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")


# mog: test run
Schedule.append(
				"python train_remote_sensing.py "
				# "python train_remote_sensing.py "
				"--compute_ndvi False --buffer False " 
                "--name nada "                
                "--images_section '' --reference_section '' "
                "--batch_size 10  --load_size 72  --crop_size 64 "
                # "--same_coordinates False "
                # "--niter 50 "

                "--save_epoch_freq 10 "
                "--use_task_loss True --use_tile_configuration True "
                # "--continue_train --epoch_count 20 "
                # "--num_threads 0 "
                "--print_freq 100 "

    #             "--source_domain Amazon_RO --stride_s 45 "
				# "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
				# "--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
				# "--source_buffer_in 2 --source_buffer_out 4 "

                "--source_domain Cerrado_MA --stride_s 19 "
                "--source_image_name_T1 MA/img_past_MA --source_image_name_T2 MA/img_present_MA "
                "--source_reference_name_T1 MA/gt_past_MA --source_reference_name_T2 MA/gt_present_MA "
                "--source_buffer_in 0  --source_buffer_out 2  "

                "--target_domain Amazon_PA  --stride_t 21 "
                "--target_image_name_T1 PA/img_past_PA  --target_image_name_T2 PA/img_present_PA "
                "--target_reference_name_T1 PA/new_gt_past_PA  --target_reference_name_T2 PA/gt_present_PA "
                "--target_buffer_in 2  --target_buffer_out 0 "

                # "--target_domain Cerrado_MA --stride_t 19 "
                # "--target_image_name_T1 MA/img_past_MA  --target_image_name_T2 MA/img_present_MA "
                # "--target_reference_name_T1 MA/gt_past_MA  --target_reference_name_T2 MA/gt_present_MA "
                # "--target_buffer_in 0  --target_buffer_out 2 "

                "--dataroot /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/")





for i in range(len(Schedule)):
    os.system(Schedule[i])


# og strides
# PA stride = 21
# RO stride = 50
# MA stride = 19

# mog: meus strides
# PA stride = 19
# RO stride = 23
# MA stride = 19

# "--source_domain Amazon_PA  --stride_s 21 "
# "--source_image_name_T1 PA/img_past_PA --source_image_name_T2 PA/img_present_PA "
# "--source_reference_name_T1 PA/new_gt_past_PA --source_reference_name_T2 PA/gt_present_PA "
# "--source_buffer_in 2 --source_buffer_out 0 "

# "--target_domain Cerrado_MA --stride_t 19 "
# "--target_image_name_T1 MA/img_past_MA --target_image_name_T2 MA/img_present_MA "
# "--target_reference_name_T1 MA/gt_past_MA --target_reference_name_T2 MA/gt_present_MA "
# "--target_buffer_in 0 --target_buffer_out 2 "

# "--source_domain Amazon_RO --stride_s 45 "
# "--source_image_name_T1 RO/img_past_RO --source_image_name_T2 RO/img_present_RO "
# "--source_reference_name_T1 RO/new_gt_past_RO --source_reference_name_T2 RO/gt_present_RO "
# "--source_buffer_in 2 --source_buffer_out 4 "


# /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/RO/
# /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/PA/new_gt_past_PA.npy
# /home/josematheus/Desktop/Mestrado/Desmatamento/code/main_data/PA/img_present_PA.npy
# 