import os
import sys
import numpy
import scipy.ndimage
import shutil
import random
import operator
from collections import OrderedDict
from PIL import Image

IMAGE_EXTS = {'png', 'jpg', 'jpeg'}
IMAGE_NAME_FMT = 'img_{0:06d}'
MIN_PIXEL_VALUE = 0.0
MAX_PIXEL_VALUE = 255.0

def get_images_recursive(base_path, subfolder):
    images = []
    subfolder_path = os.path.join(base_path, subfolder)
    for root, dirs, files in os.walk(subfolder_path):
        for filename in files:
            extension = filename.split('.')[-1].lower()
            if extension in IMAGE_EXTS:
                path = os.path.join(root, filename)
                rel_path = path[len(subfolder_path) + 1:]
                images.append(rel_path)
    return images


def index_folder(folder):
    index = dict()
    for subfolder in os.listdir(folder):
        path = os.path.join(folder, subfolder)
        if not os.path.isdir(path):
            continue
        images = get_images_recursive(folder, subfolder)
        if len(images) == 0:
            continue
        index[subfolder] = images
    return index


def extract_top_folders(index, top):
    folders_and_sizes = [(folder, len(images)) for folder, images in index.items()]
    folders_and_sizes_sorted = sorted(folders_and_sizes, key=operator.itemgetter(1), reverse=True)
    folders_and_sizes_top = folders_and_sizes_sorted[:top]
    folders_top = [folder for folder, _ in folders_and_sizes_top]
    index_top = OrderedDict()
    for category in folders_top:
        index_top[category] = index[category]
    return index_top


####  Augmentation routines start  ####


def apply_gamma_correction(image_array, min_gamma, max_gamma):
    negative_idx = image_array < 0.0
    image_array[negative_idx] = 0.0
    gamma = random.uniform(min_gamma, max_gamma)
    normalized_image = image_array / 255.0
    result = 255.0 * numpy.power(normalized_image, gamma)
    return result


def adjust_colors(image_array, color_change_limit):
    delta_r = 0.0
    delta_g = 0.0
    delta_b = 0.0
    if random.randint(0, 1) != 0:
        delta_r = random.uniform(-color_change_limit, color_change_limit)
    if random.randint(0, 1) != 0:
        delta_g = random.uniform(-color_change_limit, color_change_limit)
    if random.randint(0, 1) != 0:
        delta_b = random.uniform(-color_change_limit, color_change_limit)
    image_array[:, :, 0] += delta_r
    image_array[:, :, 1] += delta_g
    image_array[:, :, 2] += delta_b
    return image_array


def apply_blur(image_array, sigma):
    s = random.gauss(sigma, 0.4 * sigma)
    s = min(max(0.0, s), 2.0 * sigma)
    result = scipy.ndimage.filters.gaussian_filter(image_array, (s, s, 0.0), mode='constant')
    return result


def add_gaussian_noise(image_array, noise_std):
    noise = numpy.random.normal(0.0, noise_std, image_array.shape)
    image_array += noise
    return image_array


def try_augment_image(src, dst):
    try:
        image = Image.open(src).convert('RGB')
        image_array = numpy.array(image, dtype='float32')

        augmentaion_algo = random.randint(0, 3)
        if augmentaion_algo == 0:
            image_array = apply_gamma_correction(image_array, 0.33, 3.0)
        if augmentaion_algo == 1:
            image_array = adjust_colors(image_array, 64.0)
        if augmentaion_algo == 2:
            image_array = apply_blur(image_array, 1.5)
        if augmentaion_algo == 3:
            image_array = add_gaussian_noise(image_array, 32.0)

        augmented_array = numpy.clip(image_array, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
        augmented_array = numpy.asarray(augmented_array, dtype='uint8')
        augmented_image = Image.fromarray(augmented_array)
        augmented_image.save(dst)
        return True
    except Exception as e:
        print('ERROR - ', e)
        print('  Source file      :', src)
        print('  Destination file :', dst)
        return False


####  Augmentation routines finish  ####


def is_broken_image(path):
    try:
        if not os.path.exists(path):
            return True
        image = Image.open(path).convert('RGB')
        if min(image.size) == 0:
            return True
        image_array = numpy.array(image, dtype='uint8')
        if min(image_array.shape) == 0:
            return True
        return False
    except:
        return True


def copy_category(src, images, size, dst):
    num_copied = 0
    for image in images:
        if (size > 0) and (num_copied >= size):
            break
        src_path = os.path.join(src, image)
        if is_broken_image(src_path):
            continue
        ext = image.split('.')[-1].lower()
        dst_name = IMAGE_NAME_FMT.format(num_copied) + '.' + ext
        dst_path = os.path.join(dst, dst_name)
        shutil.copy(src_path, dst_path)
        num_copied += 1
    if (num_copied >= size) or (num_copied == 0):
        return num_copied, 0
    num_augmented = 0
    attempt_counter = 0
    while num_copied + num_augmented < size:
        current_idx = attempt_counter % len(images)
        current_image = images[current_idx]
        src_path = os.path.join(src, current_image)
        dst_name = IMAGE_NAME_FMT.format(num_copied + num_augmented) + '.png'
        dst_path = os.path.join(dst, dst_name)
        augmentation_result = try_augment_image(src_path, dst_path)
        if augmentation_result:
            num_augmented += 1
        attempt_counter += 1
    return num_copied, num_augmented


def copy_with_balancing(src, index, size, dst):
    num_copied = 0
    num_augmented = 0
    for category, images in index.items():
        print('  Processing category:', category)
        category_src = os.path.join(src, category)
        category_dst = os.path.join(dst, category)
        os.mkdir(category_dst)
        cat_copied, cat_augmented = copy_category(category_src, images, size, category_dst)
        print('    Copied - {:<6d},    augmented - {:<6d}'.format(cat_copied, cat_augmented))
        num_copied += cat_copied
        num_augmented += cat_augmented
    return num_copied, num_augmented


def main():
    random.seed(42)  # Gives more deterministic behavior and contains answer to the main question.

    import argparse
    parser = argparse.ArgumentParser(
        prog='Selects the most populated categories and (optionally) balances them by augmentation.',
        usage='--src <folder-with-images> --dst <folder-for-result> --top <top-categories> [--size <category-size>]')
    parser.add_argument('--src', type=str, default='', help='Source folder', dest='src')
    parser.add_argument('--top', type=int, default=0, help='', dest='top')
    parser.add_argument('--size', type=int, default=0, help='', dest='size')
    parser.add_argument('--dst', type=str, default='', help='Folder for result', dest='dst')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print('PROCESSING STOPPED - Source folder does not exist!')
        sys.exit(1)
    if args.top == 0:
        print('PROCESSING STOPPED - Parameter "top" has invalid value =', args.top)
        sys.exit(1)
    if not os.path.exists(args.dst):
        print('PROCESSING STOPPED - Destination folder does not exist!')
        sys.exit(1)
    if len(os.listdir(args.dst)) > 0:
        print('PROCESSING STOPPED - Destination folder is not empty!')
        sys.exit(1)

    print('Indexing...')
    index = index_folder(args.src)
    if len(list(index.keys())) == 0:
        print('PROCESSING STOPPED - Zero images found in source folder!')
        sys.exit(1)

    print('Extracting top categories...')
    top_index = extract_top_folders(index, args.top)
    print('  Extracted:', len(list(top_index.keys())))

    print('Balancing...')
    num_copied, num_augmented = copy_with_balancing(args.src, top_index, args.size, args.dst)

    print()
    print('Statistics:')
    print('  - Categories         :', len(list(index.keys())))
    print('  - Copied from source :', num_copied)
    print('  - Augmented          :', num_augmented)
    total = num_copied + num_augmented
    print('  - Total              :', total)
    ratio = 100.0 * num_augmented / total
    print('  - Augmented vs. total : {:<4.2f}%'.format(ratio))
    print('Processing done.')

if __name__ == '__main__':
    main()
