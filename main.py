import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import random
from numpy import linalg
from scipy import linalg
import plotly.express as px


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
    n = 10
    for t in range(var.shape[0] - n):
        alpha = var.ax[t]
        alpha_2 = var.ax[t + n]
        delta = abs(alpha_2 - alpha)
        list_2.append(delta)
        list.append(alpha)
    length = 50
    if index == 0:
        first = list_2.index(min(list_2[2 * length:])) + 42 * length
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
        var = var.iloc[first + 1000:]
    else:
        var = var.iloc[:first + 50]
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
    # ind = pd.Index([i for i in range(data.shape[0])])
    # data.set_index(ind, inplace=True)
    # for i in range(data.shape[0]):
    #     data.loc[i, 'ax'] += 395
    #     data.loc[i, 'ax'] = abs(data.loc[i, 'ax'])
    #     data.loc[i, 'ay'] -= 170
    #     data.loc[i, 'ay'] = abs(data.loc[i, 'ay'])
    #     data.loc[i, 'az'] -= 170
    #     data.loc[i, 'az'] = abs(data.loc[i, 'az'])
    # data['res'] = data.ax ** 2 + data.ay ** 2 + data.az ** 2
    data.to_csv(f'data_after_calibration/{acc1.split("/")[0]}/full_data_of_{acc1.split("/")[0]}.csv')


def check(direct, path):
    var = pd.read_csv(f'{direct}')
    list = []
    listx = []
    listy = []
    listz = []
    for t in range(var.shape[0]):
        alpha = math.degrees(math.acos(float(var.Rotation[t].split(',')[0][1:])) * 2)
        list.append(alpha)
        listx.append(float(var.Position[t].split(',')[0][1:]))
        listy.append(float(var.Position[t].split(',')[1]))
        listz.append(float(var.Position[t].split(',')[2][:-1]))
    nums = [i for i in range(len(list))]
    new = pd.read_csv(f'{path}')
    nums_2 = [i for i in range(len(new.ax))]
    fig, axs = plt.subplots(4, 1, constrained_layout=True, figsize=(16, 10))
    axs[0].plot(nums, list)
    axs[0].set_title('Угол поворота')
    axs[0].set_ylabel('Градусы')
    # axs[1].plot(nums_2, new.res)
    axs[1].plot(nums_2, new.ax)
    axs[1].set_title('Значения акселерометра по оси x')
    # axs[1].plot(nums, listx)
    # axs[2].plot(nums, listy)
    # axs[3].plot(nums, listz)
    axs[2].plot(nums_2, new.ay)
    axs[2].set_title('Значения акселерометра по оси y')
    axs[3].plot(nums_2, new.az)
    axs[3].set_title('Значения акселерометра по оси z')
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
    for t in range(1, vive.shape[0]):
        line_1 = [float(vive.Rotation[t].split(',')[0][1:]), float(vive.Rotation[t].split(',')[1]),
                  float(vive.Rotation[t].split(',')[2]),
                  float(vive.Rotation[t].split(',')[3][:-1])]
        line_2 = [float(vive.Rotation[t - 1].split(',')[0][1:]), float(vive.Rotation[t - 1].split(',')[1]),
                  float(vive.Rotation[t - 1].split(',')[2]),
                  float(vive.Rotation[t - 1].split(',')[3][:-1])]
        teta_1 = np.arctan(2 * (line_1[1] * line_1[2] + line_1[0] * line_1[3]) / (2 * (line_1[0] ** 2 + line_1[1]) - 1))
        teta_2 = np.arctan(2 * (line_2[1] * line_2[2] + line_2[0] * line_2[3]) / (2 * (line_2[0] ** 2 + line_2[1]) - 1))
        if abs(np.degrees(teta_1) - np.degrees(teta_2)) >= 10:
            # print(t)
            vive.loc[t, "Rotation"] = f'({line_1[0]},{-line_1[1]},{-line_1[2]},{-line_1[3]})'
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
    filt_acc = []
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
    files = []
    for file in os.listdir(acc):
        files.append(file)
    files.sort(key=lambda x: int(x.split('.')[0]))
    filtration_acc(f'{acc}')
    for file in files:
        var = pd.read_csv(f'{acc}/{file}', sep=',')
        line = np.matrix([-(var.ax.mean() + 395) * (2 * 9.8 / 365), (var.ay.mean() - 165) * (2 * 9.8 / 165),
                          (var.az.mean() - 165) * (2 * 9.8 / 165)])
        filt_line = np.matrix(
            [-(var.new_ax.mean() + 395) * (2 * 9.8 / 365), (var.new_ay.mean() - 165) * (2 * 9.8 / 165),
             (var.new_az.mean() - 165) * (2 * 9.8 / 165)])
        line2 = np.matrix([var.gx.mean(), var.gy.mean(), var.gz.mean()])
        matrix_acc.append(line)
        matrix_gir.append(line2)
        filt_acc.append(filt_line)
    # print(-(2 * 9.8 / 365))
    # print((2 * 9.8 / 165))
    # for el in matrix_acc:
    #     el /= np.linalg.norm(el)
    with open(f'for_formulas/{test}_test/acc/result_of_acc.dat', 'wb') as f:
        for line in matrix_acc:
            np.savetxt(f, line, fmt='%.8f')
    with open(f'for_formulas/{test}_test/acc/result_of_acc_filt.dat', 'wb') as f:
        for line in filt_acc:
            np.savetxt(f, line, fmt='%.8f')

    with open(f'for_formulas/{test}_test/gir/result_of_gir.txt', 'wb') as f:
        for line in matrix_gir:
            np.savetxt(f, line, fmt='%.8f')
    # for file in df.vive:
    #     var = pd.read_csv(f'{vive}/{file}')
    for file in files:
        var = pd.read_csv(f'{vive}/{file}', sep=',')
        vect_field = []
        all_angles = []
        for t in range(var.shape[0]):
            line = [float(var.Rotation[t].split(',')[0][1:]), float(var.Rotation[t].split(',')[1]),
                    float(var.Rotation[t].split(',')[2]),
                    float(var.Rotation[t].split(',')[3][:-1])]
            # q = Qvat(line[0], line[1], line[2], line[3])
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
        qvat = [np.cos(aver_angle / 2), aver_vec[0] * np.sin(aver_angle / 2),
                aver_vec[1] * np.sin(aver_angle / 2),
                aver_vec[2] * np.sin(aver_angle / 2)]
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
    # g = np.matrix([0, -1, 0])
    with open(f'for_formulas/{test}_test/vive/result_of_vive.txt', 'wb') as f:
        for line in matrix_vive:
            np.savetxt(f, line, fmt='%.8f')
    for i in range(len(matrix_vive)):
        # print('det = ', np.linalg.det(matrix_vive[i]))
        matrix_vect.append((matrix_vive[i] * g.T).T)
    with open(f'for_formulas/{test}_test/vive/result.dat', 'wb') as f:
        for line in matrix_vect:
            np.savetxt(f, line, fmt='%.8f')


def Qvat_app(vive, acc, type):
    matrix_acc = []
    matrix_gir = []
    matrix_gv = []
    for file in os.listdir(acc):
        var = pd.read_csv(f'{acc}/{file}', sep=',')
        line = np.matrix([-(var.ax.mean() + 395) * (2 * 9.8 / 365), (var.ay.mean() - 165) * (2 * 9.8 / 165),
                          (var.az.mean() - 165) * (2 * 9.8 / 165)])
        line2 = np.matrix([var.gx.mean(), var.gy.mean(), var.gz.mean()])
        matrix_acc.append(line)
        matrix_gir.append(line2)

    with open(f'for_formulas/qvat_app_{type}/result_of_acc.txt', 'wb') as f:
        for line in matrix_acc:
            np.savetxt(f, line, fmt='%.8f')

    with open(f'for_formulas/qvat_app_{type}/result_of_gir.txt', 'wb') as f:
        for line in matrix_gir:
            np.savetxt(f, line, fmt='%.8f')
    for file in os.listdir(vive):
        all_angles = []
        vect_field = []
        var = pd.read_csv(f'{vive}/{file}', sep=',')
        for t in range(var.shape[0]):
            line = [float(var.Rotation[t].split(',')[0][1:]), float(var.Rotation[t].split(',')[1]),
                    float(var.Rotation[t].split(',')[2]),
                    float(var.Rotation[t].split(',')[3][:-1])]
            alpha = math.acos(line[0]) * 360 / np.pi
            all_angles.append(alpha)
            v = [line[1] / np.cos(alpha / 2), line[2] / np.cos(alpha / 2), line[3] / np.cos(alpha / 2)]
            vect_field.append(v)
        for t in range(1, len(vect_field)):
            vect_field[t] /= np.linalg.norm(vect_field[t])
        aver_vec = np.mean(vect_field, axis=0)
        aver_vec /= np.linalg.norm(aver_vec)
        aver_angle = np.mean(all_angles)
        qvat = Qvat(math.cos(aver_angle / 2), aver_vec[0] * math.sin(aver_angle / 2),
                    aver_vec[1] * math.sin(aver_angle / 2),
                    aver_vec[2] * math.sin(aver_angle / 2))
        g = Qvat(0, 0, -9.8, 0)
        g_v = (qvat * g * qvat.Sopr()).ToArr()[1:]
        g_v.append(-1)
        matrix_gv.append(np.matrix(g_v))
    with open(f'for_formulas/qvat_app_{type}/result_of_vive.txt', 'wb') as f:
        for line in matrix_gv:
            np.savetxt(f, line, fmt='%.8f')


def MNK(path):
    acc_mat = []
    vive_mat = []
    with open(f'{path}/result_of_acc.txt', 'r') as f:
        for line in f:
            acc_mat.append(list(map(float, line[:-1].split(' '))))
    with open(f'{path}/result_of_vive.txt', 'r') as f:
        for line in f:
            vive_mat.append(list(map(float, line[:-1].split(' '))))
    acc_mat = np.matrix(acc_mat)
    vive_mat = np.matrix(vive_mat)
    result = ((vive_mat.T * vive_mat).getI()) * vive_mat.T * acc_mat
    with open(f'{path}/result.txt', 'wb') as f:
        for line in result:
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


def Make_plot(vive, acc):
    for file in os.listdir(acc):
        var = pd.read_csv(f'{acc}/{file}')
        vive_ = pd.read_csv(f'{vive}/{file}')
        angle = []
        for i in range(vive_.shape[0]):
            angle.append(math.degrees(math.acos(float(vive_.Rotation[i].split(',')[0][1:])) * 2))
        nums = [i + 1 for i in range(var.shape[0])]
        fig, axs = plt.subplots(4, 1, constrained_layout=True, figsize=(16, 10))
        axs[0].plot([i for i in range(vive_.shape[0])], angle)
        axs[0].set_title('Угол поворота')
        axs[1].plot(nums, var.ax)
        axs[1].set_title('Значения акселерометра по оси x')
        axs[2].plot(nums, var.ay)
        axs[2].set_title('Значения акселерометра по оси y')
        axs[3].plot(nums, var.az)
        axs[3].set_title('Значения акселерометра по оси z')
        fig.suptitle(f'{file}')
        plt.show()


def All_gr(vive, acc):
    angle = []
    ax = []
    ay = []
    az = []
    all_vive = pd.DataFrame()
    all_acc = pd.DataFrame()
    files = []
    for file in os.listdir(acc):
        files.append(file)
    files.sort(key=lambda x: int(x.split('.')[0]))
    for file in files:
        var = pd.read_csv(f'{acc}/{file}')
        vive_ = pd.read_csv(f'{vive}/{file}')
        all_vive = pd.concat([all_vive, vive_])
        all_acc = pd.concat([all_acc, var])
        for i in range(vive_.shape[0]):
            angle.append(math.degrees(math.acos(float(vive_.Rotation[i].split(',')[0][1:])) * 2))
        for i in range(var.shape[0]):
            ax.append(var.ax[i])
            ay.append(var.ay[i])
            az.append(var.az[i])
    all_vive.to_csv('for_formulas/all_vive.csv')
    all_acc.to_csv('for_formulas/all_acc.csv')
    fig, axs = plt.subplots(4, 1, constrained_layout=True, figsize=(16, 10))
    axs[0].plot([i for i in range(len(angle))], angle)
    axs[0].set_title('Угол поворота')
    axs[0].set_ylabel('Градусы')
    axs[1].plot([i for i in range(len(ax))], ax)
    axs[1].set_title('Значения акселерометра по оси x')
    axs[2].plot([i for i in range(len(ax))], ay)
    axs[2].set_title('Значения акселерометра по оси y')
    axs[3].plot([i for i in range(len(ax))], ay)
    axs[3].set_title('Значения акселерометра по оси z')
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


def Last(direct, acc):
    files = 0
    all = pd.DataFrame()
    for path in os.listdir(direct):
        for file in os.listdir(f"{direct}/{path}"):
            if file == 'LHR-9E483859.xrrelated.csv':
                var = pd.read_csv(f'{direct}/{path}/{file}', sep=';')
                if files != 24 and files != 25:
                    all = pd.concat([all, var])
                # var.to_csv(f'data_of_vive_after_all_first/{files}.csv')
                files += 1
    ind = pd.Index([i for i in range(all.shape[0])])
    all.set_index(ind, inplace=True)
    for t in range(1, all.shape[0]):
        line_1 = [float(all.Rotation[t].split(',')[0][1:]), float(all.Rotation[t].split(',')[1]),
                  float(all.Rotation[t].split(',')[2]),
                  float(all.Rotation[t].split(',')[3][:-1])]
        line_2 = [float(all.Rotation[t - 1].split(',')[0][1:]), float(all.Rotation[t - 1].split(',')[1]),
                  float(all.Rotation[t - 1].split(',')[2]),
                  float(all.Rotation[t - 1].split(',')[3][:-1])]
        teta_1 = np.arctan(2 * (line_1[1] * line_1[2] + line_1[0] * line_1[3]) / (2 * (line_1[0] ** 2 + line_1[1]) - 1))
        teta_2 = np.arctan(2 * (line_2[1] * line_2[2] + line_2[0] * line_2[3]) / (2 * (line_2[0] ** 2 + line_2[1]) - 1))
        if abs(np.degrees(teta_1) - np.degrees(teta_2)) >= 10:
            all.loc[t, "Rotation"] = f'({line_1[0]},{-line_1[1]},{-line_1[2]},{-line_1[3]})'
    all.to_csv('data_after_calibration/vive_before/Controller.csv')
    files = 0
    # acceler = []
    # for file in os.listdir(acc):
    #     acceler.append(file)
    # acceler.sort(key=lambda x: int(x.split('-')[2]))
    # for file in acceler:
    #     var = pd.read_csv(f'{acc}/{file}')
    #     os.remove(f'{acc}/{file}')
    #     var.to_csv(f'data_of_acceler_after_all_first/{files}.csv')
    #     files += 1


def Start_first():
    # Clear('data_of_vive_after_all_first')
    # Clear('data_of_acceler_after_all_first')
    # Last('data_before_calibration/vive_before', 'data_of_acceler_after_all_first')
    # Qvat_app('data_of_vive_after_all_first', 'data_of_acceler_after_all_first', 'f')
    # MNK('for_formulas/qvat_app_f')
    # Make_plot('data_of_vive_after_all_first', 'data_of_acceler_after_all_first')
    # All_gr('data_of_vive_after_all_first', 'data_of_acceler_after_all_first')
    Prepare_for_formulas('data_of_acceler_after_all_first', 'data_of_vive_after_all_first', 'first')


def Start_second():
    Change_first_ver('data_before_calibration', 'vive_after', 'acceler_after/imu_data-0-3-.csv',
                     'acceler_after/imu_data-0-7-.csv')
    # check('data_after_calibration/vive_after/Controller.csv',
    #       'data_after_calibration/acceler_after/full_data_of_acceler_after.csv')
    Clear('data_of_acceler_after_all_second')
    Clear('data_of_vive_after_all_second')
    New_approach('data_after_calibration/acceler_after/full_data_of_acceler_after.csv',
                 'data_after_calibration/vive_after/Controller.csv', 90)
    # check('data_after_calibration/vive_after/Controller.csv',
    #       'data_after_calibration/acceler_after/full_data_of_acceler_after.csv')
    # Qvat_app('data_of_vive_after_all_second', 'data_of_acceler_after_all_second', 's')
    # MNK('for_formulas/qvat_app_s')
    # Make_plot('data_of_vive_after_all_second', 'data_of_acceler_after_all_second')
    # All_gr('data_of_vive_after_all_second', 'data_of_acceler_after_all_second')
    Prepare_for_formulas('data_of_acceler_after_all_second', 'data_of_vive_after_all_second', 'second')


def All(path):
    var = pd.read_csv(path)
    teta = []
    psi = []
    for t in range(var.shape[0]):
        line = [float(var.Rotation[t].split(',')[0][1:]), float(var.Rotation[t].split(',')[1]),
                float(var.Rotation[t].split(',')[2]),
                float(var.Rotation[t].split(',')[3][:-1])]
        teta.append(
            np.degrees(np.arctan(2 * (line[1] * line[2] + line[0] * line[3]) / (2 * (line[0] ** 2 + line[1]) - 1))))
        psi.append(
            np.degrees(np.arctan(2 * (line[3] * line[2] + line[0] * line[1]) / (2 * (line[0] ** 2 + line[3]) - 1))))
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(16, 10))
    axs[0].plot([i for i in range(len(teta))], teta)
    axs[0].set_title('Угол тангажа')
    axs[0].set_ylabel('Градусы')
    axs[1].plot([i for i in range(len(teta))], psi)
    axs[1].set_title('Угол крена')
    axs[1].set_ylabel('Градусы')
    plt.show()


def Print(path):
    mat = []
    alpha_mas = []
    beta_mas = []
    gamma_mas = []
    for file in os.listdir(path):
        with open(f'{path}/{file}', 'r') as f:
            m = []
            for n, line in enumerate(f, 1):
                line = line.rstrip('\n').split(' ')
                m.append(list(map(float, line)))
                if n % 3 == 0:
                    mat.append(m.copy())
                    m.clear()
    for e in mat:
        gamma = np.degrees(np.arctan(e[0][1] / e[0][0]))
        temp = np.sin(gamma)
        beta = -np.degrees(np.arctan(e[0][2] * temp / e[0][0]))
        alpha = np.degrees(np.arctan(e[1][2] / e[2][2]))
        alpha_mas.append(alpha)
        beta_mas.append(beta)
        gamma_mas.append(gamma)
    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(16, 10))
    axs[0].plot([i for i in range(len(alpha_mas))], alpha_mas)
    axs[0].set_title('Угол тангажа')
    axs[0].set_ylabel('Градусы')
    axs[1].plot([i for i in range(len(beta_mas))], beta_mas)
    axs[1].set_title('Угол крена')
    axs[1].set_ylabel('Градусы')
    axs[2].plot([i for i in range(len(gamma_mas))], gamma_mas)
    axs[2].set_title('Угол рыскания')
    axs[2].set_ylabel('Градусы')
    plt.show()


from scipy import signal


def filt(mas):
    b, a = signal.butter(3, 0.1)
    filt_ = signal.filtfilt(b, a, mas, method="gust")
    return filt_


def solution(vive, acc):
    acc_mat = []
    vive_mat = []
    with open(acc, 'r') as f:
        for line in f:
            a = list(map(float, line.split(' ')))
            # if len(a) != 4:
            #     a.append(-1)
            acc_mat.append(a)
    with open(vive, 'r') as f:
        for line in f:
            a = list(map(float, line.split(' ')))
            if len(a) != 4:
                a.append(-1)
            vive_mat.append(a)
    acc_mat = np.matrix(acc_mat)
    vive_mat = np.matrix(vive_mat)
    result = ((vive_mat.T * vive_mat).getI()) * vive_mat.T * acc_mat
    nev = np.matrix(vive_mat * result - acc_mat).getT()
    nev = np.asarray(nev).reshape(3, acc_mat.shape[0])
    # print(nev[0])
    plt.plot([i for i in range(len(nev[0]))], nev[0], label='Ось x')
    plt.plot([i for i in range(len(nev[0]))], nev[1], label='Ось y')
    plt.plot([i for i in range(len(nev[0]))], nev[2], label='Ось z')
    plt.legend()
    plt.grid()
    plt.show()
    print(result)
    err = result[-1]
    # print('err = ', err)
    result = np.matrix(np.delete(result, [3], axis=0))
    a, b, c = linalg.svd(result)
    # print('s = ', b)
    # print('frac = ', b[0] / b[2])
    result = (result.getI())
    m = linalg.norm(result, axis=1)
    # print('mat = \n', result)
    print('m_coef = ', m)
    e_1 = result[0] / m[0]
    e = []
    e.append(np.asarray(e_1).reshape(-1))
    e_2 = np.cross(np.cross(e[0], (result[1] / m[1])), e[0])
    e.append(np.asarray(e_2).reshape(-1))
    e_3 = np.cross(e[0], e[1])
    e.append(np.asarray(e_3).reshape(-1))
    alpha = [np.degrees(np.arcsin(np.linalg.norm(np.cross(e[1], result[1] / m[1]))))]
    alpha.append(np.degrees(np.arcsin(np.linalg.norm(np.cross(e[2], result[2] / m[2])))))
    e = np.matrix(e).getT()
    e = np.asarray(e).reshape(3, 3)
    # print('e = \n', e)
    # print('alpha = ', alpha)
    gamma = np.degrees(np.arctan(e[0][1] / e[0][0]))
    temp = np.sin(gamma)
    beta = -np.degrees(np.arctan(e[0][2] * temp / e[0][0]))
    alpha = np.degrees(np.arctan(e[1][2] / e[2][2]))
    print('alpha = ', alpha)
    print('beta = ', beta)
    print('gamma = ', gamma)


def filtration_acc(path):
    for file in os.listdir(path):
        df = pd.read_csv(f'{path}/{file}')
        sig = filt(list(df['ax']))
        df['new_ax'] = sig
        sig = filt(list(df['ay']))
        df['new_ay'] = sig
        sig = filt(list(df['az']))
        df['new_az'] = sig
        df.to_csv(f'{path}/{file}')


if __name__ == '__main__':
    # Start_second()
    Start_first()
    # df = pd.read_csv('data_of_acceler_after_all_second/5.csv')
    # sig = list(df['ax'])
    # filt_sig = list(df['new_ax'])
    # fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(16, 10))
    # axs[0].plot([i for i in range(len(sig))], sig, c='gray')
    # axs[0].plot([i for i in range(len(filt_sig))], filt_sig, c='r')
    # axs[0].set_title('Значения акселерометра по оси x')
    # sig = list(df['ay'])
    # filt_sig = list(df['new_ay'])
    # axs[1].plot([i for i in range(len(sig))], sig, c='gray')
    # axs[1].plot([i for i in range(len(filt_sig))], filt_sig, c='r')
    # axs[1].set_title('Значения акселерометра по оси y')
    # sig = list(df['az'])
    # filt_sig = list(df['new_az'])
    # axs[2].plot([i for i in range(len(sig))], sig, c='gray')
    # axs[2].plot([i for i in range(len(filt_sig))], filt_sig, c='r')
    # axs[2].set_title('Значения акселерометра по оси z')
    # plt.show()
    # All('data_after_calibration/vive_after/Controller.csv')
    # All('data_after_calibration/vive_before/Controller.csv')
    # Print('for_formulas/second_test/vive')
    print('Неподвижное all: ')
    solution('for_formulas/first_test/vive/result.dat', 'for_formulas/first_test/acc/result_of_acc.dat')
    # print('Медленное: ')
    # solution('for_formulas/second_test/vive/result.dat', 'for_formulas/second_test/acc/result_of_acc.dat')
    # print('Медленное после фильтрации: ')
    # solution('for_formulas/second_test/vive/result.dat', 'for_formulas/second_test/acc/result_of_acc_filt.dat')
    # Qvat_app('data_of_vive_after_all_second', 'data_of_acceler_after_all_second', 's')
    # MNK('for_formulas/qvat_app_s')
    # Qvat_app('even/vive', 'even/acc', 'f')
    # MNK('for_formulas/qvat_app_f')
    # print('Неподвижное even: ')
    # solution('for_formulas/qvat_app_f/result_of_vive.txt', 'for_formulas/qvat_app_f/result_of_acc.txt')
    # Qvat_app('odd/vive', 'odd/acc', 'f')
    # MNK('for_formulas/qvat_app_f')
    # print('Неподвижное odd: ')
    # solution('for_formulas/qvat_app_f/result_of_vive.txt', 'for_formulas/qvat_app_f/result_of_acc.txt')
