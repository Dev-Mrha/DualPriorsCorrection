from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'facecnet_data':{
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['my_train_wraped_data'],
		'train_target_root': dataset_paths['my_train_data'],
		'train_flow_root': dataset_paths['my_train_flow'],
		'test_source_root': dataset_paths['my_test_wraped_data'],
		'test_target_root': dataset_paths['my_test_data'],
		'test_flow_root': dataset_paths['my_test_flow']
	},
	'facecnet_data_finetune':{
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['w_face'],
		'train_target_root': dataset_paths['o_face'],
		'train_flow_x_root': dataset_paths['o2s_x'],
		'train_flow_y_root': dataset_paths['o2s_y'],
		'train_ldmk_root': dataset_paths['ldmk'],
		'test_source_root': dataset_paths['test_w_face'],
		'test_target_root': dataset_paths['test_o_face'],
		'test_flow_x_root': dataset_paths['test_flow_x'],
		'test_flow_y_root': dataset_paths['test_flow_y'],
		'test_ldmk_root': dataset_paths['test_ldmk'],
	},
    'linenet_data': {
		'transforms': transforms_config.EncodeTransforms,
		'train_dir' : dataset_paths['train_dir'],
		'test_dir' : dataset_paths['test_dir']
	}
}
