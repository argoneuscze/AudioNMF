import os
import subprocess

use_advanced = False

# ===

examples_dir = '../examples'
output_dir = '../debug'

comp_map = {
    'r': 'RAW',
    'm': 'MDCT',
    's': 'STFT'
}

# ===

peaq_flag = '--basic'
if use_advanced:
    peaq_flag = '--advanced'

sample_filenames = list()

for file in os.listdir(examples_dir):
    if file.endswith('.wav'):
        sample_filenames.append(os.path.splitext(file)[0])

for fn in sample_filenames:
    orig = os.path.join(examples_dir, '{}.wav'.format(fn))

    print('== Testing {} =='.format(fn))

    # one at a time
    current_key = ['m']
    for comp in current_key:
        # for comp in comp_map.keys():
        c_file = os.path.join(output_dir, '{}_dec_anmf{}.wav'.format(fn, comp))
        peaq_out = subprocess.check_output(['peaq', peaq_flag, orig, c_file]).decode('utf-8')

        spl = peaq_out.split()
        odg = float(spl[3])
        di = float(spl[6])

        print('[{}] Objective Difference Grade: {}, Distortion Index: {}'.format(comp_map[comp], odg, di))
