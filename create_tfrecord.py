import os
import random
import tensorflow as tf
from PIL import Image
from lxml import etree
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Path to the input dataset')
FLAGS = flags.FLAGS


def create_tf_example(filename, data, label_map_dict):
    """Convert XML data to tf.Example proto."""
    height = int(data['size']['height'])
    width = int(data['size']['width'])
    filename = filename.encode('utf8')
    image_format = 'jpg'.encode('utf8')

    with tf.gfile.GFile(data['path'], 'rb') as fid:
        encoded_jpg = fid.read()

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes_text = []
    classes = []
    for obj in data['object']:
        xmins.append(float(obj['bndbox']['xmin']) / width)
        ymins.append(float(obj['bndbox']['ymin']) / height)
        xmaxs.append(float(obj['bndbox']['xmax']) / width)
        ymaxs.append(float(obj['bndbox']['ymax']) / height)
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def create_tf_record(examples, output_fn, data_dir, label_map_dict):
    """Create TFRecord files from examples."""
    output_path = os.path.join(data_dir, output_fn)

    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, fn in enumerate(examples):
        annotation_path = os.path.join(os.path.join(data_dir, 'annotations'), fn)
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = create_tf_example(fn, data, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(_):
    label_dict_path = os.path.join(FLAGS.data_dir, 'label_map.pbtxt')
    label_map_dict = label_map_util.get_label_map_dict(label_dict_path)

    annotation_path = os.path.join(FLAGS.data_dir, 'annotations')
    annotations = [x for x in os.listdir(annotation_path) if x.endswith('.xml')]
    random.shuffle(annotations)
    num_train = int(0.8 * len(annotations))
    train_examples = annotations[:num_train]
    test_examples = annotations[num_train:]

    create_tf_record(train_examples, 'train.record', FLAGS.data_dir, label_map_dict)
    create_tf_record(test_examples, 'test.record', FLAGS.data_dir, label_map_dict)

if __name__ == '__main__':
    tf.app.run()
