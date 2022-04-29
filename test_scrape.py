import os, sys
import cv2
import time
import shutil

import inference as inf


ROOT = 'data/test/scrape/'
OUT_DIR = 'testing/scrape/'

sys.stdout = open(OUT_DIR + "result.txt", 'w')

avg_time = 0
num_img = 0

for dir in os.listdir(ROOT):
    img_dir = ROOT + dir + '/'
    dir_score = 0
    for img_file in os.listdir(img_dir):
        img_path = img_dir + img_file
        img_name = img_file.split('.')[0]
        if not os.path.exists(OUT_DIR + img_name + '.png'):
            inf_time = time.time()
            preds, vis = inf.run(data=img_path, model='multi', mode='eval', output=OUT_DIR)
            inf_time = time.time() - inf_time
            avg_time += inf_time
            num_img +=1 
            if preds:
                cv2.imwrite(OUT_DIR + img_name + '.png', vis)
                print(f"Image [{img_name:>10}] : Target {dir:>10} -- Pred {preds['clss'][0]:>10}]")
                if preds['clss'][0] == dir:
                    dir_score += 1
            else:
                shutil.copy(
                    os.path.join(img_dir, img_file),
                    OUT_DIR
                )
    print(f"[{dir}] : {dir_score}/{len(os.listdir(img_dir))}")

print(f"Average inference time: {avg_time/num_img}")

sys.stdout.close()
        