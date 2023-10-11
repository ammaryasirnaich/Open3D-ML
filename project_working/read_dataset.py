# Reading and visualizing the dataset

import open3d.ml.torch as ml3d
import open3d.ml as _ml3d

import logging
import open3d.ml as _ml3d

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from open3d.ml import datasets
from open3d.ml.torch.dataloaders import TorchDataloader as Dataloader

import argparse
from tqdm import tqdm


#Read a dataset by specifying the path. We are also providing the cache directory and training split.
dataset = ml3d.datasets.KITTI(dataset_path='/workspace/data/kitti_detection/kitti', cache_dir='./logs/cache',training_split=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])
#Split the dataset for 'training'. You can get the other splits by passing 'validation' or 'test'


#view the first 1000 frames using the visualizer
# MyVis = ml3d.vis.Visualizer()
# MyVis.visualize_dataset(dataset, 'training',indices=range(100))




ObjectDetection = _ml3d.utils.get_module("pipeline", "ObjectDetection", "torch")
PointPillars = _ml3d.utils.get_module("model","PointPillars","torch")

cfg = _ml3d.utils.Config.load_from_file("/workspace/Open3D/build/Open3D-ML/ml3d/configs/pointpillars_kitti.yml")

model = PointPillars(device="cuda", **cfg.model)

pipeline = ObjectDetection(model, dataset, device="gpu")

# load the parameters.
ckpt_path ="/workspace/Open3D/build/Open3D-ML/logs/pointpillars_kitti_202012221652utc.pth"
pipeline.load_ckpt(ckpt_path=ckpt_path)

# train_split = dataset.get_split('training')
# data = train_split.get_data(0)
# # ['data']

# result = pipeline.run_inference(data)

# # data = test_split[5]['data']

# boxes = data['bounding_boxes']
# boxes.extend(result)

vis = Visualizer()

lut = LabelLUT()
for val in sorted(dataset.label_to_names.keys()):
    lut.add_label(val, val)

# # Uncommenting this assigns bbox color according to lut
# # for key, val in sorted(dataset.label_to_names.items()):
# #     lut.add_label(key, val)

# vis.visualize([{
#     "name": 'Kitti',
#     'points': data['point']
# }],
#                 lut,
#                 bounding_boxes=boxes)


boxes = []
data_list = []
test_split = dataset.get_split('training')
for idx in tqdm(range(100)):
    
    data = test_split.get_data(idx)
    

    result = pipeline.run_inference(data)[0]

    boxes = data['bounding_boxes']
    boxes.extend(result)

    data_list.append({
        "name": "Kitti" + '_' + str(idx),
        'points': data['point'],
        'bounding_boxes': boxes
    })

# vis.visualize(data_list, lut, bounding_boxes=boxes)
vis.visualize(data_list, lut)










'''
Running on test data split
'''

'''
boxes = []
data_list = []
test_split = dataset.get_split('test')
for idx in tqdm(range(100)):
    
    data = test_split.get_data(idx)
    

    result = pipeline.run_inference(data)[0]

    boxes = data['bounding_boxes']
    boxes.extend(result)

    data_list.append({
        "name": "Kitti" + '_' + str(idx),
        'points': data['point'],
        'bounding_boxes': boxes
    })

vis.visualize(data_list, lut, bounding_boxes=boxes)


'''