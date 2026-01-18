import os
import csv

def get_paired_paths(lq_folder, gt_folder, filename_tmpl='{}'):
    lq_files = sorted([f for f in os.listdir(lq_folder) if os.path.isfile(os.path.join(lq_folder, f))])
    gt_files = sorted([f for f in os.listdir(gt_folder) if os.path.isfile(os.path.join(gt_folder, f))])
    paired = []
    for lq, gt in zip(lq_files, gt_files):
        lq_path = os.path.join(lq_folder, lq)
        gt_path = os.path.join(gt_folder, gt)
        paired.append({'lq_path': lq_path, 'gt_path': gt_path})
    return paired

def save_paired_paths_to_csv(paired_paths, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['lq_path', 'gt_path'])
        writer.writeheader()
        for item in paired_paths:
            writer.writerow(item)

if __name__ == '__main__':

    lq_folder = 'datasets/AODLibpro_img/MixLib_32401l40i/hybrid/ab'  
    gt_folder = 'datasets/AODLibpro_img/MixLib_32401l40i/hybrid/gt'  
    csv_path = 'datasets/AODLibpro_img/MixLib_32401l40i/hybrid/meta_info.csv'  
    
    print('=== generate lq/gt paired paths ===')
    print(f'lq_folder: {lq_folder}')
    print(f'gt_folder: {gt_folder}')
    print(f'ouput csv file: {csv_path}')
    

    if not os.path.exists(lq_folder):
        print(f'error: lq_folder "{lq_folder}" does not exist')
        exit(1)
    
    if not os.path.exists(gt_folder):
        print(f'error: gt_folder "{gt_folder}" does not exist')
        exit(1)
    

    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        try:
            os.makedirs(csv_dir)
            print(f'output dir has been created: {csv_dir}')
        except Exception as e:
            print(f'error: output dir fails to be created "{csv_dir}": {e}')
            exit(1)
    
    try:
        paired_paths = get_paired_paths(lq_folder, gt_folder)
        save_paired_paths_to_csv(paired_paths, csv_path)
        print(f'Complete! {len(paired_paths)} has been saved in {csv_path}')
    except Exception as e:
        print(f'error: something went wrong: {e}')
        exit(1) 