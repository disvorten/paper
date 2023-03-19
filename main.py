import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import random
import plotly
import plotly.graph_objects as go
from statistics import median
from mpl_toolkits.mplot3d import Axes3D


class Qvat:
    def __init__(self, q_0=0.0, q_1=0.0, q_2=0.0, q_3=0.0):
        self.q_0 = float(q_0)
        self.q_1 = float(q_1)
        self.q_2 = float(q_2)
        self.q_3 = float(q_3)

    def __mul__(self, right):
        a_0 = self.q_0 * right.q_0 - self.q_1 * right.q_1 - self.q_2 * right.q_2 - self.q_3 * right.q_3
        a_1 = self.q_0 * right.q_1 + self.q_1 * right.q_0 + self.q_2 * right.q_3 - self.q_3 * right.q_2
        a_2 = self.q_0 * right.q_2 - self.q_1 * right.q_3 + self.q_2 * right.q_0 + self.q_3 * right.q_1
        a_3 = self.q_0 * right.q_3 + self.q_1 * right.q_2 - self.q_2 * right.q_1 + self.q_3 * right.q_0
        return Qvat(a_0, a_1, a_2, a_3)

    def Sopr(self):
        return Qvat(self.q_0, -self.q_1, -self.q_2, -self.q_3)

    def __repr__(self):
        print(f"[{self.q_0},{self.q_1},{self.q_2},{self.q_3}]")

    def __str__(self):
        return f"[{self.q_0},{self.q_1},{self.q_2},{self.q_3}]"

    def ToArr(self):
        return [self.q_0, self.q_1, self.q_2, self.q_3]


def Change_Acc_first_ver(name, acc, index):
    var = pd.read_csv(f'{name}/{acc}', sep=',')
    list = []
    list_2 = []
    n = 20
    for t in range(var.shape[0] - n):
        alpha = var.ax[t]
        alpha_2 = var.ax[t + n]
        delta = abs(alpha_2 - alpha)
        list_2.append(delta)
        list.append(alpha)
    length = 330
    if index == 0:
        first = list_2.index(min(list_2[4 * length:])) + 10 * length
    else:
        first = list_2.index(min(list_2[length:-length])) + length
    if index == 0:
        first += n
        t0 = float(str(var.server_time[first].split(':')[-1])[0:4])
        m0 = float(str(var.server_time[first].split(':')[-2]))
    else:
        first -= n
        t0 = float(str(var.server_time[first].split(':')[-1])[0:4])
        m0 = float(str(var.server_time[first].split(':')[-2]))
    if index == 0:
        var = var.iloc[first:]
    else:
        var = var.iloc[:first]
    return var, t0, m0


def Change_first_ver(name, direct, acc1, acc2):
    for file in os.listdir(f'{name}/{direct}'):
        g = Qvat(0, 0, -9.8, 0)
        k = 0
        new_1, t0, m0 = Change_Acc_first_ver(name, acc1, 0)
        new_2, t1, m1 = Change_Acc_first_ver(name, acc2, 1)
        data = new_1
        var = pd.read_csv(f'{name}/{direct}/{file}', sep=';')
        while float(str(var.Timestamp[k].split(':')[-1])[0:4]) != t0 or float(
                str(var.Timestamp[k].split(':')[-2])) != m0:
            k += 1
        first = k
        while float(str(var.Timestamp[k].split(':')[-1])[0:4]) != t1 or float(
                str(var.Timestamp[k].split(':')[-2])) != m1:
            k += 1
        second = k
        var = var.iloc[first:second]
        var.to_csv(f'data_after_calibration/{direct}/{file}')
    files = [acc1.split("/")[1], acc2.split("/")[1]]
    for file in os.listdir(f'{name}/{acc1.split("/")[0]}'):
        if file not in files:
            data = pd.concat([data, pd.read_csv(f'{name}/{acc1.split("/")[0]}/{file}', sep=',')])
    data = pd.concat([data, new_2])
    data.to_csv(f'data_after_calibration/{acc1.split("/")[0]}/full_data_of_{acc1.split("/")[0]}.csv')


def check(direct, path):
    var = pd.read_csv(f'{direct}')
    list = []
    for t in range(var.shape[0]):
        alpha = math.degrees(math.acos(float(var.Rotation[t].split(',')[0][1:])) * 2)
        list.append(alpha)
    nums = [i for i in range(len(list))]
    new = pd.read_csv(f'{path}')
    nums_2 = [i for i in range(len(new.ax))]
    fig, axs = plt.subplots(4, 1, constrained_layout=True, figsize=(16, 10))
    axs[0].plot(nums, list)
    axs[1].plot(nums_2, new.ax)
    axs[2].plot(nums_2, new.ay)
    axs[3].plot(nums_2, new.az)
    plt.show()


def Making_Smaller(var, var_, files):
    k = 0
    ind = pd.Index([i for i in range(var.shape[0])])
    var.set_index(ind, inplace=True)
    ind = pd.Index([i for i in range(var_.shape[0])])
    var_.set_index(ind, inplace=True)
    second = 0
    for t in range(1, var_.shape[0]):
        alpha = math.acos(float(var_.Rotation[t].split(',')[0][1:])) * 2
        alpha_2 = math.acos(float(var_.Rotation[t - 1].split(',')[0][1:])) * 2
        delta = (alpha_2 - alpha) * 180 / np.pi
        if abs(delta) >= 10:
            var_.iloc[:t].to_csv(f'vive_clear/{files}.csv')
            t0 = float(str(var_.Timestamp[t].split(':')[-1])[0:5])
            m0 = float(str(var_.Timestamp[t].split(':')[-2]))
            while float(str(var.server_time[k].split(':')[-1])[0:5]) != t0 or float(
                    str(var.server_time[k].split(':')[-2])) != m0:
                k += 1
            var.iloc[:k].to_csv(f'acc_clear/{files}.csv')
            while abs(delta) >= 10:
                t += 1
                alpha = math.acos(float(var_.Rotation[t].split(',')[0][1:])) * 2
                alpha_2 = math.acos(float(var_.Rotation[t - 1].split(',')[0][1:])) * 2
                delta = (alpha_2 - alpha) * 180 / np.pi
            second = t
            t1 = float(str(var_.Timestamp[second].split(':')[-1])[0:5])
            m1 = float(str(var_.Timestamp[second].split(':')[-2]))
            while float(str(var.server_time[k].split(':')[-1])[0:5]) != t1 or float(
                    str(var.server_time[k].split(':')[-2])) != m1:
                k += 1
            break
    if second != 0:
        Making_Smaller(var.iloc[k:], var_.iloc[second:], files + 1)
    else:
        var_.to_csv(f'vive_clear/{files}.csv')
        var.to_csv(f'acc_clear/{files}.csv')


def New_approach(aceler, vive_, N):
    acc = pd.read_csv(f'{aceler}', sep=',')
    vive = pd.read_csv(f'{vive_}', sep=',')
    angle = []
    for t in range(1, vive.shape[0]):
        alpha = math.degrees(math.acos(float(vive.Rotation[t].split(',')[0][1:])) * 2)
        alpha_2 = math.degrees(math.acos(float(vive.Rotation[t - 1].split(',')[0][1:])) * 2)
        delta = alpha_2 - alpha
        if abs(delta) >= 10:
            line = [float(vive.Rotation[t].split(',')[0][1:]), float(vive.Rotation[t].split(',')[1]),
                    float(vive.Rotation[t].split(',')[2]),
                    float(vive.Rotation[t].split(',')[3][:-1])]
            vive.loc[t, "Rotation"] = f'({-line[0]},{-line[1]},{-line[2]},{-line[3]})'
            alpha = math.degrees(math.acos(float(vive.Rotation[t].split(',')[0][1:])) * 2)
        angle.append(alpha)
    # plt.plot([i for i in range(len(angle))], angle)
    # plt.show()
    vive.to_csv(f'{vive_}')
    length = acc.shape[0] // N
    length_ = vive.shape[0] // N
    for i in range(N):
        data_of_vive = vive[i * length_:length_ * (i + 1)]
        data_of_acceler = acc[i * length:length * (i + 1)]
        data_of_acceler.to_csv(f'data_of_acceler_after_all_second/{i}.csv')
        data_of_vive.to_csv(f'data_of_vive_after_all_second/{i}.csv')


def Create_data_second_test(acceler, vive):
    var = pd.read_csv(f'{acceler}', sep=',')
    var_ = pd.read_csv(f'{vive}', sep=',')
    # waste_time = 1
    # waste_time = 0
    # all_waste = []
    # all_index = []
    # all_ind_acc = []
    # count = 0
    # length = 0
    # for t in range(var_.shape[0] - 1):
    #     alpha = math.acos(float(var_.Rotation[t].split(',')[0][1:])) * 2
    #     alpha_2 = math.acos(float(var_.Rotation[t + 1].split(',')[0][1:])) * 2
    #     delta = (alpha_2 - alpha) * 180 / np.pi
    #     # print(delta)
    #     if abs(delta) >= 10:
    #         waste_time += 1
    #         # print(delta)
    # print(waste_time)
    # for t in range(var_.shape[0] - 1):
    #     alpha = math.acos(float(var_.Rotation[t].split(',')[0][1:])) * 2
    #     alpha_2 = math.acos(float(var_.Rotation[t + 1].split(',')[0][1:])) * 2
    #     delta = (alpha_2 - alpha) * 180 / np.pi
    #     if abs(delta) >= 10:
    #         print('first = ', delta, t)
    #         var_.iloc[:t].to_csv(f'vive_clear/{files}.csv')
    #         files += 1
    #         first = t
    #         count = 1
    #         length += 1
    #     if count == 1 and abs(delta) < 1:
    #         print('second = ', delta, t)
    #         count = 0
    #         second = t
    #         all_waste.append([i for i in range(first - length + 1, second)])
    #         length = 0
    # print(all_waste)
    # for elem in all_waste:
    #     for x in elem:
    #         all_index.append(x)
    #     k = 0
    #     t0 = float(str(var_.Timestamp[elem[0]].split(':')[-1])[0:5])
    #     m0 = float(str(var_.Timestamp[elem[0]].split(':')[-2]))
    #     t1 = float(str(var_.Timestamp[elem[-1] + 1].split(':')[-1])[0:5])
    #     m1 = float(str(var_.Timestamp[elem[-1] + 1].split(':')[-2]))
    #     while float(str(var.server_time[k].split(':')[-1])[0:5]) != t0 and int(
    #             str(var.server_time[k].split(':')[-2])) == m0:
    #         k += 1
    #     first = k
    #     while float(str(var.server_time[k].split(':')[-1])[0:5]) != t1 and int(
    #             str(var.server_time[k].split(':')[-2])) == m1:
    #         k += 1
    #     if waste_time == 1:
    #         var_.iloc[:elem[0] - 1000].to_csv('vive_clear/0.csv')
    #         var_.iloc[elem[-1] + 500:].to_csv('vive_clear/1.csv')
    #         var.iloc[:first - 6].to_csv('acc_clear/0.csv')
    #         var.iloc[first + k + 6:].to_csv('acc_clear/1.csv')
    #         break
    #     for i in range(first, first + k):
    #         all_ind_acc.append(i)
    # print(all_index)
    # var_.drop(index=all_index, inplace=True)
    # var.drop(labels=all_ind_acc, inplace=True)
    Making_Smaller(var, var_, 0)
    # var_ = pd.read_csv('vive_clear/0.csv')
    # for x in os.listdir('vive_clear'):
    #     gr = []
    #     new = pd.read_csv(f'vive_clear/{x}')
    #     for t in range(new.shape[0] - 1):
    #         alpha = math.acos(float(new.Rotation[t].split(',')[0][1:])) * 2
    #         alpha_2 = math.acos(float(new.Rotation[t + 1].split(',')[0][1:])) * 2
    #         delta = (alpha_2 - alpha) * 180 / np.pi
    #         gr.append(delta)
    #     # plt.plot([i for i in range(len(gr))], gr)
    #     # plt.title(f'vive_clear/{x}')
    #     # plt.show()
    #     if x != '0.csv':
    #         var_ = pd.concat([var_, pd.read_csv(f'vive_clear/{x}')])
    # var = pd.read_csv('acc_clear/0.csv')
    # for x in os.listdir('acc_clear'):
    #     if x != '0.csv':
    #         var = pd.concat([var, pd.read_csv(f'acc_clear/{x}')])
    ind = pd.Index([i for i in range(var.shape[0])])
    var.set_index(ind, inplace=True)
    ind = pd.Index([i for i in range(var_.shape[0])])
    var_.set_index(ind, inplace=True)
    # var_.to_csv('vive_clear/final.csv')
    # var.to_csv('acc_clear/final.csv')
    # var_ = pd.read_csv('vive_clear/final.csv')
    # var = pd.read_csv('acc_clear/final.csv')
    # length = var.shape[0] // N
    # length_ = var_.shape[0] // N
    # for i in range(N):
    #     data_of_vive = var_[i * length_:length_ * (i + 1)]
    #     data_of_acceler = var[i * length:length * (i + 1)]
    #     data_of_acceler.to_csv(f'data_of_acceler_after_all_second/{i}.csv')
    #     data_of_vive.to_csv(f'data_of_vive_after_all_second/{i}.csv')


def Prepare_for_formulas(acc, vive, test):
    matrix_acc = []
    matrix_gir = []
    matrix_vive = []
    g = np.matrix([0, -9.8, 0])
    random.seed(a=200)
    accel = []
    vive_ = []
    for file in os.listdir(acc):
        accel.append(file)
    for file in os.listdir(vive):
        vive_.append(file)
    index = [i for i in range(len(accel))]
    random.shuffle(index)
    df = pd.DataFrame(index=index)
    df['acc'] = accel
    df['vive'] = vive_
    df.sort_index(inplace=True)
    all_acc = []
    # for file in df.acc:
    #     var = pd.read_csv(f'{acc}/{file}')
    for file in os.listdir(acc):
        var = pd.read_csv(f'{acc}/{file}', sep=',')
        line = np.matrix([var.ax.mean() / 10.300522410768254, var.ay.mean() / 5.9310844102049325,
                          var.az.mean() / 7.510015646104054])
        line2 = np.matrix([var.gx.mean(), var.gy.mean(), var.gz.mean()])
        all_acc.append([var.ax.mean() / 10.300522410768254, var.ay.mean() / 5.9310844102049325,
                        var.az.mean() / 7.510015646104054])
        matrix_acc.append(line)
        matrix_gir.append(line2)

    with open(f'for_formulas/{test}_test/acc/result_of_acc.dat', 'wb') as f:
        for line in matrix_acc:
            np.savetxt(f, line, fmt='%.8f')

    with open(f'for_formulas/{test}_test/gir/result_of_gir.txt', 'wb') as f:
        for line in matrix_gir:
            np.savetxt(f, line, fmt='%.8f')
    # for file in df.vive:
    #     var = pd.read_csv(f'{vive}/{file}')
    for file in os.listdir(vive):
        var = pd.read_csv(f'{vive}/{file}', sep=',')
        vect_field = []
        all_angles = []
        for t in range(var.shape[0]):
            line = [float(var.Rotation[t].split(',')[0][1:]), float(var.Rotation[t].split(',')[1]),
                    float(var.Rotation[t].split(',')[2]),
                    float(var.Rotation[t].split(',')[3][:-1])]
            q = Qvat(line[0], line[1], line[2], line[3])
            alpha = math.acos(line[0]) * 360 / np.pi
            # if count != 0 and abs(alpha - last_angle) >= 30:
            #     alpha = 360 - alpha
            all_angles.append(alpha)
            v = [line[1] / np.cos(alpha / 2), line[2] / np.cos(alpha / 2), line[3] / np.cos(alpha / 2)]
            vect_field.append(v)
        for t in range(1, len(vect_field)):
            vect_field[t] /= np.linalg.norm(vect_field[t])
        aver_vec = np.mean(vect_field, axis=0)
        aver_vec /= np.linalg.norm(aver_vec)
        aver_angle = np.mean(all_angles)
        qvat = [math.cos(aver_angle / 2), aver_vec[0] * math.sin(aver_angle / 2),
                aver_vec[1] * math.sin(aver_angle / 2),
                aver_vec[2] * math.sin(aver_angle / 2)]
        matrix_vive.append(qvat_to_matrix(qvat))
    # fig, axs = plt.subplots(4, 1, constrained_layout=True, figsize=(16, 10))
    # axs[0].plot([i for i in range(len(all))], all)
    # axs[1].plot([i for i in range(len(all_acc))], [all_acc[i][0] for i in range(len(all_acc))])
    # axs[2].plot([i for i in range(len(all_acc))], [all_acc[i][1] for i in range(len(all_acc))])
    # axs[3].plot([i for i in range(len(all_acc))], [all_acc[i][2] for i in range(len(all_acc))])
    #     ax = plt.axes(projection='3d')
    #     for i in range(len(vect_field)):
    #         ax.quiver(0, 0, 0, vect_field[i][0], vect_field[i][1], vect_field[i][2])
    #     ax.quiver(0, 0, 0, 0, -9.8, 0, color='g')
    #     ax.quiver(0, 0, 0, aver_vec[0], aver_vec[1], aver_vec[2], color='r')
    #     plt.show()
    # vect_field.append(sum)
    # fig = go.Figure(data=[go.Scatter3d(x=[vect_field[i][0] for i in range(len(vect_field))],
    #                                    y=[vect_field[i][1] for i in range(len(vect_field))],
    #                                    z=[vect_field[i][2] for i in range(len(vect_field))])])
    # fig.show()
    matrix_vect = []
    with open(f'for_formulas/{test}_test/vive/result_of_vive.txt', 'wb') as f:
        for line in matrix_vive:
            np.savetxt(f, line, fmt='%.8f')
    for i in range(len(matrix_vive)):
        matrix_vect.append((matrix_vive[i] * g.T).T)
    with open(f'for_formulas/{test}_test/vive/result.dat', 'wb') as f:
        for line in matrix_vect:
            np.savetxt(f, line, fmt='%.8f')


def qvat_to_matrix(tur):
    w = tur[0]
    x = tur[1]
    y = tur[2]
    z = tur[3]
    return np.matrix([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (z * x + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (z * y - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])


def Make_plot(direct):
    for file in os.listdir(direct):
        var = pd.read_csv(f'{direct}/{file}')
        nums = [i + 1 for i in range(var.shape[0])]
        fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(16, 10))
        axs[0].plot(nums, var.ax)
        axs[1].plot(nums, var.ay)
        axs[2].plot(nums, var.az)
        plt.show()


def Make_Clear(path, acc, N):
    count_files = 0
    count = 0
    for x in os.listdir(path):
        if pd.read_csv(f'{path}/{x}').shape[0] < 500:
            os.remove(f'{path}/{x}')
            os.remove(f'{acc}/{x}')
        else:
            count_files += 1
    data_acc = pd.DataFrame()
    data_vive = pd.DataFrame()
    # mas = []
    for x in os.listdir(path):
        new = pd.read_csv(f'{path}/{x}')
        var = pd.read_csv(f'{acc}/{x}')
        data_acc = pd.concat([data_acc, var])
        data_vive = pd.concat([data_vive, new])
    ind = pd.Index([i for i in range(data_acc.shape[0])])
    data_acc.set_index(ind, inplace=True)
    ind = pd.Index([i for i in range(data_vive.shape[0])])
    data_vive.set_index(ind, inplace=True)
    # vect_field = []
    # for t in range(1, len(data_vive)):
    #     line = [float(data_vive.Rotation[t].split(',')[0][1:]), float(data_vive.Rotation[t].split(',')[1]),
    #             float(data_vive.Rotation[t].split(',')[2]),
    #             float(data_vive.Rotation[t].split(',')[3][:-1])]
    #     alpha = math.acos(line[0]) * 2
    #     alpha_2 = math.acos(float(data_vive.Rotation[t - 1].split(',')[0][1:])) * 2
    #     delta = (alpha_2 - alpha) * 180 / np.pi
    #     v = [line[1] / np.cos(alpha / 2), line[2] / np.cos(alpha / 2), line[3] / np.cos(alpha / 2)]
    #     vect_field.append(v)
    #     if abs(delta) >= 10:
    #         length = data_vive.shape[0]
    #         while t < length:
    #             line = [float(data_vive.Rotation[t].split(',')[0][1:]), float(data_vive.Rotation[t].split(',')[1]),
    #                     float(data_vive.Rotation[t].split(',')[2]),
    #                     float(data_vive.Rotation[t].split(',')[3][:-1])]
    #             alpha = math.acos(line[0]) * 2
    #             # v = [line[1] / np.cos(alpha / 2), line[2] / np.cos(alpha / 2), line[3] / np.cos(alpha / 2)]
    #             # vect_field.append(v)
    #             data_vive.loc[t, "Rotation"] = f'({line[0]},{-line[1]},{-line[2]},{-line[3]})'
    #             line = [float(data_vive.Rotation[t].split(',')[0][1:]), float(data_vive.Rotation[t].split(',')[1]),
    #                     float(data_vive.Rotation[t].split(',')[2]),
    #                     float(data_vive.Rotation[t].split(',')[3][:-1])]
    #             t += 1
    #         break
    mas = []
    angle = []
    for t in range(1, len(data_vive)):
        alpha = math.acos(float(data_vive.Rotation[t].split(',')[0][1:])) * 2
        angle.append(alpha * 180 / np.pi)
        alpha_2 = math.acos(float(data_vive.Rotation[t - 1].split(',')[0][1:])) * 2
        delta = (alpha_2 - alpha) * 180 / np.pi
        mas.append(delta)
    for t in range(1, new.shape[0]):
        alpha = math.acos(float(new.Rotation[t].split(',')[0][1:])) * 2
        alpha_2 = math.acos(float(new.Rotation[t - 1].split(',')[0][1:])) * 2
        delta = (alpha_2 - alpha) * 180 / np.pi
        mas.append(delta)
    # plt.plot([i for i in range(len(mas))], mas)
    plt.plot([i for i in range(len(angle))], angle)
    # ax = plt.axes(projection='3d')
    # for i in range(len(vect_field)):
    #     ax.quiver(0, 0, 0, vect_field[i][0], vect_field[i][1], vect_field[i][2])
    # plt.show()
    length = var.shape[0] * count_files // N
    length_ = new.shape[0] * count_files // N
    for i in range(1, N // count_files + 1):
        data_of_vive = new[(i - 1) * length_:length_ * i]
        data_of_acceler = var[(i - 1) * length:length * i]
        data_of_acceler.to_csv(f'data_of_acceler_after_all_second/{i + count * (N // count_files)}.csv')
        data_of_vive.to_csv(f'data_of_vive_after_all_second/{i + count * (N // count_files)}.csv')
    count += 1


def Clear(folder):
    for file in os.listdir(folder):
        os.remove(f'{folder}/{file}')


def Last(direct):
    files = 0
    for path in os.listdir(direct):
        for file in os.listdir(f"{direct}/{path}"):
            if file == 'LHR-9E483859.xrrelated.csv':
                var = pd.read_csv(f'{direct}/{path}/{file}', sep=';')
                var.to_csv(f'data_of_vive_after_all_first/{files}.csv')
                files += 1


def Start_first():
    Clear('data_of_vive_after_all_first')
    Last('data_before_calibration/vive_before')
    Prepare_for_formulas('data_of_acceler_after_all_first', 'data_of_vive_after_all_first', 'first')


def Start_second():
    Change_first_ver('data_before_calibration', 'vive_after', 'acceler_after/imu_data-0-1-.csv',
                     'acceler_after/imu_data-0-4-.csv')
    Clear('data_of_acceler_after_all_second')
    Clear('data_of_vive_after_all_second')
    New_approach('data_after_calibration/acceler_after/full_data_of_acceler_after.csv',
                 'data_after_calibration/vive_after/Controller.csv', 15)
    check('data_after_calibration/vive_after/Controller.csv',
          'data_after_calibration/acceler_after/full_data_of_acceler_after.csv')
    # Clear('vive_clear')
    # Clear('acc_clear')
    # Create_data_second_test('data_after_calibration/acceler_after/full_data_of_acceler_after.csv',
    #                         'data_after_calibration/vive_after/Controller.csv')
    # Make_Clear('vive_clear', 'acc_clear', 20)
    Prepare_for_formulas('data_of_acceler_after_all_second', 'data_of_vive_after_all_second', 'second')


if __name__ == '__main__':
    Start_second()
    # Start_first()
