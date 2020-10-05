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

	# def get_img(self, idx, raw_img=False):
	# 	raise NotImplementedError

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
		try:
			if len(self.last_mod) < 2:
				img1id, modid, img2id = self.get_2nd_training_query()
			else:
				img1id, modid, img2id = self.get_1st_training_query()
		except:
			img1id, modid, img2id = self.get_1st_training_query()

		out = {}
		out['source_img_id'] = img1id
		out['source_img_data'] = self.get_scene(img1id)
		out['target_img_id'] = img2id
		out['target_img_data'] = self.get_scene(img2id)
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

		# list of all possible attributes
		# TODO : include position in attributes
		# TODO : see if 'cylinder' needs to be replaced by 'triangle'. Because query text uses triangle in place of cylinder
		attributes = ["small", "large", "gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan", "rubber", "metal", "cube", "sphere", "cylinder"]

		# creating a dict to map each attribute to its index in attributes list
		attr2idx = { attr:attributes.index(attr) for attr in attributes }

		# creating initial node representation for each object in scene
		# making multi-hot vectors of objects in src scene
		object_nodes_in_scene = []

		for obj in scene['objects']:
			obj_rep = [0]*len(attributes)
			obj_rep[attr2idx[obj['shape']]] = 1
			obj_rep[attr2idx[obj['size']]] = 1
			obj_rep[attr2idx[obj['color']]] = 1
			obj_rep[attr2idx[obj['material']]] = 1

			object_nodes_in_scene.append(obj_rep)

		# creating adj matrices for each relation type
		relation_types = ['left', 'right', 'front', 'behind']

		adj_matrices = { relation : np.zeros((number_of_objects_in_scene,number_of_objects_in_scene)) for relation in relation_types }

		for obj_index in range(number_of_objects_in_scene):
			for relation in relation_types:
				for related_obj in scene['relationships'][relation][obj_index]:
					adj_matrices[relation][obj_index,related_obj] = 1

		out['objects'] = np.array(object_nodes_in_scene)
		out['relations'] = adj_matrices

		return out