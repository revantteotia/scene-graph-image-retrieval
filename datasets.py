"""Provides data for training and testing."""
import numpy as np
import PIL
import skimage.io
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random

# temp list of empty images : to ignore images with no objects
lis=[('test', 9832), ('train', 1565), ('test', 4132), ('train', 2881), ('train', 1161), ('train', 9313), ('test', 17076), ('train', 4363), ('train', 1221), ('test', 7235), ('test', 12701), ('train', 2446), ('train', 9837), ('test', 11441), ('test', 1349), ('train', 3023), ('train', 14329), ('test', 3918), ('train', 11780), ('train', 4729), ('train', 15256), ('train', 6460), ('test', 5471), ('test', 12318), ('train', 2821), ('test', 11197), ('test', 1100), ('test', 14355), ('train', 3155), ('train', 7540), ('test', 18320), ('train', 6668), ('train', 16116), ('test', 5346), ('test', 1275), ('train', 1087), ('test', 6223), ('train', 10547), ('test', 13383)]

class BaseDataset(torch.utils.data.Dataset):
	"""Base class for a dataset."""

	def __init__(self):
		super(BaseDataset, self).__init__()
		self.imgs = []
		self.test_queries = []

	def get_loader(self,
					batch_size,
					shuffle=False,
					drop_last=False,
					num_workers=0):
		return torch.utils.data.DataLoader(
			self,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			drop_last=drop_last,
			collate_fn=lambda i: i)

	def get_test_queries(self):
		return self.test_queries

	def get_all_texts(self):
		raise NotImplementedError

	def __getitem__(self, idx):
		return self.generate_random_query_target()

	def generate_random_query_target(self):
		raise NotImplementedError

	def get_img(self, idx, raw_img=False):
		raise NotImplementedError

	def get_scene(self, idx, raw_img=False):
		raise NotImplementedError

class CSSDataset(BaseDataset):
	"""CSS dataset."""

	def __init__(self, path, split='train', transform=None):
		super(CSSDataset, self).__init__()

		self.scene_path = path + '/scenes'
		self.img_path = path + '/images'

		self.transform = transform
		self.split = split
		self.data = np.load(path + '/css_toy_dataset_novel2_small.dup.npy', allow_pickle=True, encoding="latin1").item()
		self.mods = self.data[self.split]['mods']
		self.imgs = []
		for objects in self.data[self.split]['objects_img']:
			label = len(self.imgs)
			if 'labels' in self.data[self.split]:
				label = self.data[self.split]['labels'][label]
			self.imgs += [{
				'objects': objects,
				'label': label,
				'captions': [str(label)]
			}]

		self.imgid2modtarget = {}
		for i in range(len(self.imgs)):
			self.imgid2modtarget[i] = []
		for i, mod in enumerate(self.mods):
			for k in range(len(mod['from'])):
				f = mod['from'][k]
				t = mod['to'][k]
				self.imgid2modtarget[f] += [(i, t)]

		self.generate_test_queries_()

	def generate_test_queries_(self):
		test_queries = []
		for mod in self.mods:
			for i, j in zip(mod['from'], mod['to']):
				test_queries += [{
					'source_img_id': i,
					'target_caption': self.imgs[j]['captions'][0],
					'mod': {
						'str': mod['to_str']
					}
				}]
		self.test_queries = test_queries

	def get_1st_training_query(self):
		i = np.random.randint(0, len(self.mods))
		mod = self.mods[i]
		j = np.random.randint(0, len(mod['from']))
		self.last_from = mod['from'][j]
		self.last_mod = [i]
		return mod['from'][j], i, mod['to'][j]

	def get_2nd_training_query(self):
		modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
		while modid in self.last_mod:
			modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
		self.last_mod += [modid]
		# mod = self.mods[modid]
		return self.last_from, modid, new_to

	def generate_random_query_target(self):
		while(True): # temp while loop to ignore empty images
			try:
				if len(self.last_mod) < 2:
					img1id, modid, img2id = self.get_2nd_training_query()
				else:
					img1id, modid, img2id = self.get_1st_training_query()
			except:
				img1id, modid, img2id = self.get_1st_training_query()
			
			if self.split=='train':
				tup1=('train',img1id)
				if tup1 not in lis:
					tup2=('train',img2id)
					if tup2 not in lis:
						break		
			elif self.split=='test':
				tup1=('test',img1id)
				if tup1 not in lis:
					tup2=('test',img2id)
					if tup2 not in lis:
						break	

		out = {}
		out['source_img_id'] = img1id
		out['source_scene_data'] = self.get_scene(img1id)
		out['source_img_data'] = self.get_img(img1id)
		out['target_img_id'] = img2id
		out['target_scene_data'] = self.get_scene(img2id)
		out['target_img_data'] = self.get_img(img2id)
		out['mod'] = {'id': modid, 'str': self.mods[modid]['to_str']}
		return out

	def __len__(self):
		return len(self.imgs)

	def get_all_texts(self):
		return [mod['to_str'] for mod in self.mods]

	def get_scene(self, idx, raw_img=False, get_2d=False):
		"""
		Gets CSS image scene graph from given image index.

		Returns :
				node features : list of multi-hot vectors for each objet in the scene
				adjacency matrices : dict of adj matrices for each relation type
		"""

		out = {}

		scene_path = self.scene_path + ('/css_%s_%06d.json' % (self.split, int(idx)))

		# loading scene json
		with open(scene_path) as fp:
			scene = json.load(fp)

		number_of_objects_in_scene = len(scene['objects'])

		attribute_types = ['shape', 'size', 'color', 'material']
		attributes = ["small", "large", "gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan", "rubber", "metal", "cube", "sphere", "cylinder"]

		# creating a dict to map each attribute to its index in attributes list
		attr2idx = { attr:attributes.index(attr) for attr in attributes }

		# creating initial node representation for each object in scene
		# making multi-hot vectors of objects in src scene
		object_nodes_in_scene = []
		positions=["top-left", "middle-left", "bottom-left", "top-center", "middle-center", "bottom-center", "top-right", "middle-right", "bottom-right"]
		
		for obj in scene['objects']:
			obj_rep = [0]*(len(attributes)+len(positions))
			lis_pos=obj['pixel_coords']
			x_coord=lis_pos[0]
			y_coord=lis_pos[1]
			x_coord_index=int((x_coord/180)*3)
			y_coord_index=int((y_coord/120)*3)
			ind=3*x_coord_index+y_coord_index
			obj_rep[len(attributes)+ind]=1

			for attr_type in attribute_types:
				obj_rep[attr2idx[obj[attr_type]]] = 1	
			
			object_nodes_in_scene.append(obj_rep)

		# converting to np array
		object_nodes_in_scene = torch.Tensor(object_nodes_in_scene)

		# creating adj matrices for each relation type
		relation_types = ['left', 'right', 'front', 'behind']

		from_nodes = []
		to_nodes = []
		edge_type = []
		
		for obj_index in range(number_of_objects_in_scene):
			for relation in relation_types:
				for related_obj in scene['relationships'][relation][obj_index]:
					from_nodes.append(obj_index)
					to_nodes.append(related_obj)
					edge_type.append( relation_types.index(relation))
		

		edge_index = torch.LongTensor([from_nodes, to_nodes])
		edge_type = torch.LongTensor(edge_type)
		
		out['objects'] = object_nodes_in_scene
		out['edge_index'] = edge_index
		out['edge_type'] = edge_type

		return out


	def get_img(self, idx, raw_img=False, get_2d=False):
		def generate_2d_image(objects):
			img = np.ones((64, 64, 3))
			colortext2values = {
				'gray': [87, 87, 87],
				'red': [244, 35, 35],
				'blue': [42, 75, 215],
				'green': [29, 205, 20],
				'brown': [129, 74, 25],
				'purple': [129, 38, 192],
				'cyan': [41, 208, 208],
				'yellow': [255, 238, 51]
			}
			for obj in objects:
				s = 4.0
				if obj['size'] == 'large':
					s *= 2
				c = [0, 0, 0]
				for j in range(3):
					c[j] = 1.0 * colortext2values[obj['color']][j] / 255.0
				y = obj['pos'][0] * img.shape[0]
				x = obj['pos'][1] * img.shape[1]
				if obj['shape'] == 'rectangle':
					img[int(y - s):int(y + s), int(x - s):int(x + s), :] = c
				if obj['shape'] == 'circle':
					for y0 in range(int(y - s), int(y + s) + 1):
						x0 = x + (abs(y0 - y) - s)
						x1 = 2 * x - x0
						img[y0, int(x0):int(x1), :] = c
				if obj['shape'] == 'triangle':
					for y0 in range(int(y - s), int(y + s)):
						x0 = x + (y0 - y + s) / 2
						x1 = 2 * x - x0
						x0, x1 = min(x0, x1), max(x0, x1)
						img[y0, int(x0):int(x1), :] = c
			return img

		if self.img_path is None or get_2d:
			img = generate_2d_image(self.imgs[idx]['objects'])
		else:
			img_path = self.img_path + ('/css_%s_%06d.png' % (self.split, int(idx)))
			with open(img_path, 'rb') as f:
				img = PIL.Image.open(f)
				img = img.convert('RGB')

		if raw_img:
			return img
		if self.transform:
			img = self.transform(img)
		return img

