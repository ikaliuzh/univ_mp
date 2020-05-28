import matplotlib.pyplot as plt
from cycler import cycler
from utils import *


color_c = cycler('color', ['k'])
style_c = cycler('linestyle', ['-', '--', ':', '-.'])
markr_c = cycler('marker', ['', '.', 'o'])
c_cms = color_c * markr_c * style_c
c_csm = color_c * style_c * markr_c


# ----------------------------------------------------------
# simulation outcome visualisation
# ----------------------------------------------------------
# restore the experiment outcomes
in_file, data = open("protocol.dat"), []
# wdata, ddata, wstardata = [], [], []
# nvars = int(in_file.readline())
npoints = 0
for item in in_file:
    temp = item.split(" ")
    temp = list(map(float, temp))
    data.append(temp)
    # wdata.append(temp[: nvars])
    # ddata.append(temp[nvars])
    # wstardata.append(temp[nvars + 1 :])
    npoints += 1
    del temp
in_file.close()
tdata = [ic for ic in range(npoints)]  # data for abscissa axis

# handle the experiment outcomes
print("Choose a variant of the graphic")
while True:
    print(
        "Input data variants in format '1'(N+) | 2NN |'e'"
        " where N ::= '1' | .. | '9'\n"
        "and 'e' requires to exit"
    )
    var = input("Input: ")
    if var[0] in "12e" and \
            all(map(lambda x : x in "123456789", var[1: ])):
        break
    else:
        print(
            "Error! "
            "Repeat choosing a variant of the graphic"
        )
if var == "e":
    pass
else:
    var = list(map(int, var))
    vars, ngraphs = var[1: ], len(data[0])
    if var[0] == 1:
        if len(vars) == 0 or \
                any(map(lambda x : x > ngraphs, vars)):
            print(
                "Error! Data "
                "variant refers to a non-existent data")
            pass
        else:
            plt.rc('axes', prop_cycle=c_cms)
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            ax.set(xlim=(0, npoints), ylim=(-0.02, 1.02))
            ax.set_aspect(500)
            ax.grid()
            ax.set_xlabel(
                input_non_empty("Label of x-axis: ")
            )
            ax.set_ylabel(
                input_non_empty("Label of y-axis: ")
            )
            plt.tight_layout(pad=1)
            for v in vars:
                xdata = []
                xdata = [item[v - 1] for item in data]
                ax.plot(tdata, xdata)
                del xdata
            # plt.text(850,0.35, "$W(0)$")
            # plt.text(850,0.6, "$W(1)$")
            plt.show()
    else:  # var[0] == 2
        if len(vars) != 2 or \
                any(map(lambda x : x > ngraphs, vars)):
            print(
                "Error! Data "
                "variant refers to a non-existent data")
            pass
        else:
            plt.rc('axes', prop_cycle=c_cms)
            fig = plt.figure(figsize=(5, 4.5))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.tick_params(axis='x',
                top=False, bottom=False,
                labelbottom=False)
            ax2.tick_params(axis='y',
                left=False, right=True,
                labelleft=False, labelright=True)
            ax1.set_ylabel(
                input_non_empty(
                    "Label of y-axis for the top graph: "
                )
            )
            ax2.set_xlabel(
                input_non_empty("Label of x-axis: ")
            )
            ax2.set_ylabel(
                input_non_empty(
                    "Label of y-axis for the bottom graph: "
                )
            )
            for ax in fig.axes:
                ax.set(xlim=(0, npoints), ylim=(-0.02, 1.02))
                ax.set_aspect(500)
                ax.grid()
            plt.tight_layout(pad=1, h_pad=-0.6)
            x1data = [item[vars[0] - 1] for item in data]
            x2data = [item[vars[1] - 1] for item in data]
            ax1.plot(tdata, x1data)
            ax2.plot(tdata, x2data)
            # plt.text(850,0.17, "$W^*(1)$")
            # plt.text(850,0.75, "$W^*(0)$")
            plt.show()
