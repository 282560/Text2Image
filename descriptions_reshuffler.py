'''
    This script should be placed in: {project_catalog}\datasets\{dataset_catalog}\text_c10.
    The main purpose of this script is shuffling descriptions in case when there are few same descriptions in row.
    Example:
        desc1
        desc1
        desc1
        desc2
        desc2
        desc3
        Turns into:
        desc1
        desc2
        desc3
        desc1
        desc2
        desc1
'''

import os


useful = 0
useless = 0
af_counter = 0
for subdir, dirs, files in os.walk('.'):
    for file in files:
        if (file == os.path.basename(__file__)) or (file == 'full_corpus.txt'):
            useless += 1
        else:
            f = open(os.path.join(subdir, file), 'r')
            lines = f.readlines()
            f.close()

            open(os.path.join(subdir, file), 'w').close()

            f = open(os.path.join(subdir, file), 'w')
            dict = {i: lines.count(i) for i in lines}

            all_vals = 0
            for k, v in dict.items():
                all_vals += int(v)

            counter = all_vals
            while True:
                for k, v in dict.items():
                    if int(v) > 0:
                        f.write(k)
                        dict[k] = int(v) - 1
                        counter -= 1
                if counter == 0:
                    break
            useful += 1
            f.close()
        af_counter += 1

print('Important files:', useful)
print('Exceptions files:', useless)
print('All files:', af_counter)