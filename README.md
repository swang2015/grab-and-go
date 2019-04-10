# Smart retail checkout

A 2018 LD hackathon project. Built with TF object detetion API. Start from: [Training a pet detector](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md).

# Directory structure

```
- data/
-- annotations/
-- images/
-- label_map.pbtxt
- model/
-- config
-- pretrained/
-- train/
-- eval/
```

# Prepare dataset

- Take photos or scrape images (the process can be adapted to any specific set of images by implememting transfer learning)
- Label the images using [LabelImg](https://github.com/tzutalin/labelImg)
- Create TFRecord train/test dataset

`python create_tfrecord.py --data_dir=<data_dir>`

# Transfer learning

- Configure training pipeline

Pick up a pretrained model from [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and edit [the corresponding config file](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs). See [tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

- Train the model

Either [on cloud](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md) or [locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md).

- Export the model

# Detection

