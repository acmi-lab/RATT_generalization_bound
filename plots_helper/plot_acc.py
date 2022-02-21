from matplotlib import pyplot as plt
import numpy as np
import argparse

def read_file(filename): 
    data = []
    with open(filename, 'r') as f:
        f.readline()
        for lines in f:
            tokens  = lines.split(',')
            tokens = [float(x) for x in tokens]
            data.append(tokens)
   
    return np.array(data)

plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here


parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('--plot-name', default="CIFAR10", type=str, help='Plot name')
parser.add_argument('--log-file', type=str, help='Log file')


args = parser.parse_args()

file_data = read_file(args.log_file)

fig = plt.figure()


l = 2.0
fc=20


plt.plot( file_data[:, 0], file_data[:, 2] , linewidth=l, color='crimson')
plt.plot( file_data[:, 0], file_data[:, 3] , linestyle = "--", linewidth=l, color='blue')


plt.ylabel('Accuracy', fontsize=fc)
plt.xlabel('Training steps', fontsize=fc)
plt.xticks(fontsize=fc)
plt.yticks(fontsize=fc)

plt.grid()
plt.title(f"{args.plot_name}",fontsize=fc)
plt.legend()

plt.savefig(f'{args.plot_name}.pdf',bbox_inches='tight', transparent=True)