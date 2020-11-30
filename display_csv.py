import matplotlib.pyplot as plt

import csv

x = []
y1 = []
y2 = []
single = True
f_name = 'gd_loss_v.csv'

with open(f_name, 'r', newline='') as f:
    plots = csv.reader(f, delimiter=';')
    for row in plots:
        x.append(float(row[0]))
        y1.append(float(row[1]))
        y2.append(float(row[2]))

if(single):
    rounding = 2

    fig, axs = plt.subplots(2)

    axs[0].set_title('Dyskryminator')
    axs[0].plot(x, y1, '-ko')
    axs[0].set(xlabel='Epoki', ylabel='Blad')

    d_maximum = max(y1)
    d_maximum_idx = y1.index(max(y1))
    d_x_max = x[d_maximum_idx]
    d_minimum = min(y1)
    d_minimum_idx = y1.index(min(y1))
    d_x_min = x[d_minimum_idx]

    middle1 = ((max(y1) - min(y1)) / 2) + min(y1)
    middle2 = ((max(y1) - min(y1)) / 2) + min(y1) + (0.2 * ((max(y1) - min(y1)) / 2))
    middle3 = ((max(y1) - min(y1)) / 2) + min(y1) - (0.2 * ((max(y1) - min(y1)) / 2))

    axs[0].plot(d_x_max, d_maximum, 'ro')
    axs[0].plot(d_x_min, d_minimum, 'ro')
    axs[0].annotate('Maks. ' + str(round(d_maximum, rounding)) + ' (' + str(int(d_x_max)) + ')', xy=(d_x_max, d_maximum), xytext=(d_x_max, middle1), arrowprops=dict(arrowstyle='->', color='red'), color='r')
    axs[0].annotate('Min. ' + str(round(d_minimum, rounding)) + ' (' + str(int(d_x_min)) + ')', xy=(d_x_min, d_minimum), xytext=(d_x_min, middle1), arrowprops=dict(arrowstyle='->', color='red'), color='r')

    axs[0].plot(x[0], y1[0], 'bo')
    axs[0].plot(x[-1], y1[-1], 'bo')
    axs[0].annotate('Pierwszy ' + str(round(y1[0], rounding)), xy=(x[0], y1[0]), xytext=(x[0], middle2), arrowprops=dict(arrowstyle='->', color='blue'), color='b')
    axs[0].annotate('Ostatni ' + str(round(y1[-1], rounding)), xy=(x[-1], y1[-1]), xytext=(x[-1], middle3), arrowprops=dict(arrowstyle='->', color='blue'),  color='b')

    axs[1].set_title('Generator')
    axs[1].plot(x, y2, '-ko')
    axs[1].set(xlabel='Epoki', ylabel='Blad')

    g_maximum = max(y2)
    g_maximum_idx = y2.index(max(y2))
    g_x_max = x[g_maximum_idx]
    g_minimum = min(y2)
    g_minimum_idx = y2.index(min(y2))
    g_x_min = x[g_minimum_idx]

    middle1 = ((max(y2) - min(y2)) / 2) + min(y2)
    middle2 = ((max(y2) - min(y2)) / 2) + min(y2) - (0.2 * ((max(y2) - min(y2)) / 2))
    middle3 = ((max(y2) - min(y2)) / 2) + min(y2) + (0.2 * ((max(y2) - min(y2)) / 2))

    axs[1].plot(g_x_max, g_maximum, 'ro')
    axs[1].plot(g_x_min, g_minimum, 'ro')
    axs[1].annotate('Maks. ' + str(round(g_maximum, rounding)) + ' (' + str(int(g_x_max)) + ')', xy=(g_x_max, g_maximum), xytext=(g_x_max, middle1), arrowprops=dict(arrowstyle='->', color='red'), color='r')
    axs[1].annotate('Min. ' + str(round(g_minimum, rounding)) + ' (' + str(int(g_x_min)) + ')', xy=(g_x_min, g_minimum), xytext=(g_x_min, middle1), arrowprops=dict(arrowstyle='->', color='red'), color='r')

    axs[1].plot(x[0], y2[0], 'bo')
    axs[1].plot(x[-1], y2[-1], 'bo')
    axs[1].annotate('Pierwszy ' + str(round(y2[0], rounding)), xy=(x[0], y2[0]), xytext=(x[0], middle2), arrowprops=dict(arrowstyle='->', color='blue'), color='b')
    axs[1].annotate('Ostatni ' + str(round(y2[-1], rounding)), xy=(x[-1], y2[-1]), xytext=(x[-1], middle3), arrowprops=dict(arrowstyle='->', color='blue'), color='b')

else:
    plt.plot(x, y1, label='Dyskryminator')
    plt.plot(x, y2, label='Generator')
    plt.legend()

plt.show()