from PIL import Image

import os


def extract_idx(name, stat1='image_', stat2='.txt'):
    new_name = name.replace(stat1, '')
    new_name = new_name.replace(stat2, '')
    new_name = new_name.replace('0', '')
    return new_name

def complete_name(idx, ext):
    num = idx
    positions = []
    while num != 0:
        positions.append(num % 10)
        num = num // 10
    complete = ((5 - len(positions)) * '0')
    for i in reversed(positions):
        complete = complete + str(i)
    return 'image_' + complete + '.' + ext

images = '102flowers'
classes_desc = 'text_c10'
new_dir = 'processed_102flowers'
new_dir_desc = 'processed_text_c10'

classes = {}
all_files = 0
for subdir, dirs, files in os.walk(classes_desc):

    if subdir != classes_desc:

        splitted = os.path.split(subdir)
        classes[splitted[1]] = []

        for file in files:
            filename, ext = os.path.splitext(file)
            tpl = (os.path.join(classes_desc, splitted[1] , filename + '.txt'), os.path.join(images, filename + '.jpeg'))
            classes[splitted[1]].append(tpl)
            all_files += 1

iter = 1
while iter <= all_files:
    to_remove = []
    for k, v in classes.items():
        if not os.path.isdir(os.path.join(new_dir_desc, k)):
            os.mkdir(os.path.join(new_dir_desc, k))
        if len(v) > 0:
            item = v.pop(0)
            (txt, img) = item

            im = Image.open(img)
            im.save(os.path.join(new_dir, complete_name(iter, 'JPEG')))

            f = open(txt, 'r')
            lines = f.readlines()
            f.close()
            f = open(os.path.join(new_dir_desc, k, complete_name(iter, 'txt')), 'w')
            for it in lines:
                f.write(it)
            f.close()

            print(iter)
            iter += 1
        else:
            print(k, 'removed.')
            to_remove.append(k)

    for it in to_remove:
        del classes[it]
    to_remove.clear()

    print('Comparing', iter, 'and', all_files)