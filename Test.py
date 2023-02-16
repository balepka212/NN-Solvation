import os
import shutil


def main():
    paths = ['LinNet', 'ResNet']
    for path in paths:
        the_path = os.path.join('/Users/balepka/PycharmProjects/msuAI/Run_results/', path)
        for filename in os.listdir(the_path):
            if filename.startswith('.'):
                continue
            dir = os.path.join(the_path, filename, 'best')
            if not os.path.isdir(dir):
                dst_path = os.path.join('/Users/balepka/PycharmProjects/msuAI/Run_results/', path, filename, 'best')
                os.makedirs(dst_path)
                srs_path = os.path.join('/Users/balepka/PycharmProjects/msuAI/Runs/', path, filename, 'best', 'best_val_model.pt')
                dst_path = os.path.join(dst_path, 'best_val_model.pt')
                shutil.copy(srs_path, dst_path)

# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
