import os
import sys
import argparse
import random
import datetime
import subprocess

IMAGE_EXTS = {'jpg', 'jpeg', 'png'}
CONVERT_IMAGESET_REL_PATH = 'build/tools/convert_imageset'
COMPUTE_IMAGE_MEAN_REL_PATH = 'build/tools/compute_image_mean'


def check_args(args):
    if not os.path.exists(args.src):
        print('Cannot find source folder:', args.src)
        print('Termination.')
        sys.exit(1)
    if not os.path.exists(args.dst):
        print('Creating new folder:', args.dst)
        os.makedirs(args.dst)
    if len(os.listdir(args.dst)) > 0:
        print('Destination folder is not empty.')
        print('Termination for safety reasons.')
        sys.exit(1)
    if not os.path.exists(args.caffe):
        print('Cannot find Caffe root:', args.caffe)
        print('Termination.')
        sys.exit(1)
    if args.size == 0:
        print('Wrong resize size:', args.size)
        print('Termination.')
        sys.exit(1)
    if (args.split <= 0.0) or (args.split >= 1.0):
        print('Wrong split ratio', args.split, ', it should be in (0.0, 1.0).')
        print('Termination.')
        sys.exit(1)


def get_images_recursive(folder, base_path):
    images = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            extension = filename.split('.')[-1].lower()
            if extension in IMAGE_EXTS:
                path = os.path.join(root, filename)
                rel_path = path[len(base_path) + 1:]
                images.append(rel_path)
    return images


def index_folder(src, minimal_size):
    index = dict()
    n_categories = 0
    n_images = 0
    categories = os.listdir(src)
    for category in categories:
        category_subfolder = os.path.join(src, category)
        if not os.path.isdir(category_subfolder):
            continue
        category_images = get_images_recursive(category_subfolder, src)
        if len(category_images) < minimal_size:
            print('  Warning! Category', category, 'has', len(category_images), 'images which is too few.')
            continue
        index[category] = category_images
        n_categories += 1
        n_images += len(category_images)
    print('  Found', n_images, ' images in', n_categories, 'categories.')
    return index


def separate_train_and_val(index, split_fraction):
    train = dict()
    val = dict()
    for category, category_images in index.items():
        val_size = int(len(category_images) * split_fraction + 0.5)
        val_size = max(val_size, 1)
        shuffled_images = category_images.copy()
        random.shuffle(shuffled_images)
        train_images = shuffled_images[val_size:]
        val_images = shuffled_images[:val_size]
        train[category] = train_images
        val[category] = val_images
    return train, val


def generate_listfile(index, listfile):
    category_label = 0
    with open(listfile, 'w') as f:
        for category in sorted(index.keys()):
            category_label_str = str(category_label)
            category_images = index[category]
            for image in category_images:
                line = '{:} {:}\n'.format(image, category_label_str)
                f.write(line)
            category_label += 1
    print('  Generated:', listfile)


def generate_lmdb(caffe_root, listfile, src, dst, db_name, size, store_png):
    db_path = os.path.join(dst, db_name)
    convert_imageset_path = os.path.join(caffe_root, CONVERT_IMAGESET_REL_PATH)
    parameters = convert_imageset_path + ' '
    parameters += '--resize_width ' + str(size) + ' --resize_height ' + str(size) + ' --shuffle '
    if store_png:
        parameters += '--encoded --encode_type png '
    parameters += src + '/ '
    parameters += listfile + ' '
    parameters += db_path
    print('  SUBPROCESS LAUNCH WITH:', parameters)
    subprocess.call(parameters, stdout=sys.stdout, stderr=sys.stdout, shell=True)


def generate_mean_binaryproto(caffe_root, dst, db_name):
    db_path = os.path.join(dst, db_name)
    mean_binaryproto_path = os.path.join(dst, 'mean.binaryproto')
    compute_image_mean_path = os.path.join(caffe_root, COMPUTE_IMAGE_MEAN_REL_PATH)
    parameters = compute_image_mean_path + ' '
    parameters += db_path + ' ' + mean_binaryproto_path
    print('SUBPROCESS LAUNCH WITH:', parameters)
    subprocess.call(parameters, stdout=sys.stdout, stderr=sys.stdout, shell=True)


def save_params_and_categories(dst, args, index):
    generation_params_path = os.path.join(dst, 'generation_params.log')
    with open(generation_params_path, 'w') as f:
        f.write('Created on: ' + str(datetime.datetime.now()) + '\n')
        f.write('\n')
        f.write('Source           : ' + args.src + '\n')
        f.write('Destination      : ' + args.dst + '\n')
        f.write('Caffe root       : ' + args.caffe + '\n')
        f.write('Resize size      : ' + str(args.size) + '\n')
        f.write('Minimal category : ' + str(args.min) + '\n')
        f.write('Split ratio      : ' + str(args.split) + '\n')
        f.write('Keep as PNG      : ' + str(args.store_png) + '\n')
        f.write('\n')
        f.write('Categories       : ' + str(len(index.keys())))
    categories_list = os.path.join(dst, 'categories.txt')
    with open(categories_list, 'w') as f:
        categories = sorted(index.keys())
        for category in categories:
            f.write(category + '\n')


def main():
    random.seed(42)  # Gives more deterministic behavior and contains answer to the main question.

    parser = argparse.ArgumentParser(
        prog='Converts an organized set of images to LMDB using folder structure to determine category.',
        usage='--src <folder-with-images> --dst <folder-to-have-LMDBS> --caffe <folder-with-caffe> '
              '--resize <size> [--split <ratio>] [--png]')
    parser.add_argument('--src', type=str, default='', help='Folder with images', dest='src')
    parser.add_argument('--dst', type=str, default='', help='Folder for LMDBs', dest='dst')
    parser.add_argument('--caffe', type=str, default='', help='Caffe root folder', dest='caffe')
    parser.add_argument('--resize', type=int, default=0, help='Image sizes inside LMDBs', dest='size')
    parser.add_argument('--min', type=int, default=1, help='Minimal number of images in category', dest='min')
    parser.add_argument('--split', type=float, default=0.1, help='How many images does to validation', dest='split')
    parser.add_argument('--png',
        default=False,
        action='store_true',
        help='Determines LMDB format (compressed PNG images or Caffe blobs)',
        dest='store_png')
    args = parser.parse_args()
    check_args(args)
    src_folder = args.src
    if src_folder.endswith('/'):
        src_folder = src_folder[:len(src_folder) - 1]

    print('Indexing folder', src_folder, '...')
    index = index_folder(src_folder, args.min)

    print('Separating train and test...')
    train, val = separate_train_and_val(index, args.split)

    print('Generate listfiles...')
    train_listfile = os.path.join(args.dst, 'train_listfile.txt')
    generate_listfile(train, train_listfile)
    val_listfile = os.path.join(args.dst, 'val_listfile.txt')
    generate_listfile(val, val_listfile)

    print('Generate LMDBs by Caffe convert_imageset...')
    generate_lmdb(args.caffe, train_listfile, src_folder, args.dst, 'train', args.size, args.store_png)
    generate_lmdb(args.caffe, val_listfile, src_folder, args.dst, 'val', args.size, args.store_png)

    print('Generate mean.binaryproto...')
    generate_mean_binaryproto(args.caffe, args.dst, 'train')

    print('Save params and categories...')
    save_params_and_categories(args.dst, args, index)

    print('Done!')

if __name__ == '__main__':
    main()
