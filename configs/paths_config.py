import os

class ModelPath():
	def __init__(self, args):
		super(ModelPath, self).__init__()
		print('Loading Model Path')
		self.args = args
		stylegan_ffhq_path = os.path.join(args.pretrained_path, 'stylegan2-ffhq-config-f.pt')
		ir_se50_path = os.path.join(args.pretrained_path, 'model_ir_se50.pth')
		shape_predictor_path = os.path.join(args.pretrained_path, 'shape_predictor_68_face_landmarks.dat')
		moco_path = os.path.join(args.pretrained_path, 'moco_v2_800ep_pretrain.pth')
		face_parsing_path = os.path.join(args.pretrained_path, 'face_parsing_model.pth')
		self.model_paths = {
			'stylegan_ffhq': stylegan_ffhq_path,
			'ir_se50': ir_se50_path,
			'shape_predictor': shape_predictor_path,
			'moco': moco_path,
			'face_parsing': face_parsing_path
		}

	def get_model_path(self):
		return self.model_paths


dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': '',
	'celeba_test': '',

	#  Cars Dataset (In the paper: Stanford cars)
	'cars_train': '',
	'cars_test': '',

	#  Horse Dataset (In the paper: LSUN Horse)
	'horse_train': '',
	'horse_test': '',

	#  Church Dataset (In the paper: LSUN Church)
	'church_train': '',
	'church_test': '',

	#  Cats Dataset (In the paper: LSUN Cat)
	'cats_train': '',
	'cats_test': ''
}

model_paths = {
	'stylegan_ffhq': 'autodl-tmp/pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'autodl-tmp/pretrained_models/model_ir_se50.pth',
	'shape_predictor': 'autodl-tmp/pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': 'autodl-tmp/pretrained_models/moco_v2_800ep_pretrain.pth'
}
