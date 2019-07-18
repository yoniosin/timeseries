import numpy as np
import pickle
import mat4py

import gdown
import re
from util.Subject import subject_generator, Subject
from util.config import load_mat, get_roi_md


class TrialData:
    def __init__(self, protocol_path, subject_type='3d'):
        def create_subject():
            def download_bold_mat(download=True):
                if download:
                    gdown.download(new_link, 'bold_mat.mat', quiet=False)
                mat = load_mat('bold_mat.mat')
                s_id = re.search(r'(Subject\d*)', open('subject_name.txt', 'r').read()).group(1)

                return mat, s_id

            needed = re.search(r'file/d/(.*)/view', link).group(1)
            if needed not in downloaded:
                new_link = ''.join(['https://drive.google.com/uc?id=', needed, '&export=download'])

                if link in open('downloaded.txt', 'r'):
                    return

                bold_mat, subject_id = download_bold_mat()
                print('download complete')
                protocol = protocols[subject_id]
                subject: Subject = subject_generator(subject_id, protocol, bold_mat, subject_type)
                pickle.dump(subject, open(f'../Data/3D/{subject.name}.pkl', 'wb'))
                print(f'Created {subject_id}')
                self.shared_min = min(self.shared_min, subject.meta_data.min_w)
                open('downloaded.txt', 'a').write(link)

        with open('downloaded.txt', 'r') as d:
            downloaded = [re.search(r'd/(.*)/view', down).group(1) for down in d]
        self.shared_min = np.inf
        protocols = mat4py.loadmat(protocol_path)['Protocols']
        keep_elements = ['subject', 'TestWatchOnset', 'TestWatchDuration', 'TestRegulateOnset', 'TestRegulateDuration']
        p_per_subject = zip(*(map(lambda a: protocols[a], keep_elements)))
        protocols = {protocol[0]: protocol[1:] for protocol in p_per_subject}
        Subject.voxels_md = get_roi_md()

        for link in open('download_links.txt', 'r'):
            try:
                create_subject()
            except:
                print(f'failed creating {link}')
                open('failed.txt', 'a').write(link)

        # with open('meta.txt', 'wb') as jf:
        #     json.dump({'min_w': self.shared_min}, jf)


def load_trial_data():
    return pickle.load(open("raw_data/trial_data.pckl", "rb"))


if __name__ == '__main__':
    data = TrialData('../raw_data/protocols.mat')
    # pickle.dump(data, open('raw_data/trial_data.pckl', 'wb'))
