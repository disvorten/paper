import numpy as np
import pandas as pd
import os


def Change_Acc(name, acc, index):
    var = pd.read_csv(f'{name}/{acc}', sep=',')
    if index == 0:
        max_gy = abs(var[:var.shape[0] // 2].gy).max()
    else:
        max_gy = abs(var[var.shape[0] // 2:].gy).max()
    for i in range(var.shape[0]):
        if abs(var.gy[i]) == max_gy:
            t = str(var.server_time[i].split(':')[-1])[0:5]
            m = str(var.server_time[i].split(':')[-2])
            if index == 0:
                new_var = var.iloc[i + 5:]
            else:
                new_var = var.iloc[:i - 5]
            break
    return t, m, new_var


def Change(name, direct, acc1, acc2):
    for file in os.listdir(f'{name}/{direct}'):
        t0, m0, new_1 = Change_Acc(name, acc1, 0)
        t1, m1, new_2 = Change_Acc(name, acc2, 1)
        data = new_1
        var = pd.read_csv(f'{name}/{direct}/{file}', sep=';')
        var.to_csv(f'data_after_calibration/{direct}/{file}')
        # first = 0
        # for i in range(var.shape[0]):
        #     if float(str(var.Timestamp[i].split(':')[-1])[0:4]) == round(float(t0), 1) and int(
        #             str(var.Timestamp[i].split(':')[-2])) == int(m0):
        #         first = i
        #     if float(str(var.Timestamp[i].split(':')[-1])[0:4]) == round(float(t1), 1) and int(
        #             str(var.Timestamp[i].split(':')[-2])) == int(m1):
        #         new_var = var.iloc[first + 5: i - 5]
        #         break
        # new_var.to_csv(f'data_after_calibration/{direct}/{file}')
    files = [acc1.split("/")[1], acc2.split("/")[1]]
    for file in os.listdir(f'{name}/{acc1.split("/")[0]}'):
        if file not in files:
            data = pd.concat([data, pd.read_csv(f'{name}/{acc1.split("/")[0]}/{file}', sep=',')])
    data = pd.concat([data, new_2])
    data.to_csv(f'data_after_calibration/{acc1.split("/")[0]}/full_data_of_{acc1.split("/")[0]}.csv')


def Create_data_first_test(acceler, vive, N):
    var = pd.read_csv(f'{acceler}', sep=',')
    files = 0
    length = 0
    for i in range(N):
        j = 0
        k = 0
        t0 = float(str(var.server_time[length].split(':')[-1])[0:5])
        t1 = round((t0 + 10.0) % 60, 2)
        m0 = int(str(var.server_time[length].split(':')[-2]))
        m1 = int(m0 + (t0 + 10.0) // 60)
        while float(str(var.server_time[j + length].split(':')[-1])[0:5]) != t1:
            j += 1
        data_of_acceler = var[length:length + j]
        length += j
        t2 = float(str(var.server_time[length].split(':')[-1])[0:5])
        t3 = round((t2 + 10.0) % 60, 2)
        while float(str(var.server_time[length + k].split(':')[-1])[0:5]) != t3:
            k += 1
        length += k
        first = 0
        var_ = pd.read_csv(f'{vive}', sep=',')
        for v in range(var_.shape[0]):
            if float(str(var_.Timestamp[v].split(':')[-1])[0:4]) == round(t0, 1) and int(
                    str(var_.Timestamp[v].split(':')[-2])) == m0:
                first = v
            if float(str(var_.Timestamp[v].split(':')[-1])[0:4]) == round(t1, 1) and int(
                    str(var_.Timestamp[v].split(':')[-2])) == m1:
                data_of_vive = var_.iloc[first: v]
                break
        data_of_acceler.to_csv(f'data_of_acceler_after_all_first/{files}.csv')
        data_of_vive.to_csv(f'data_of_vive_after_all_first/{files}.csv')
        files += 1


def Create_data_second_test(acceler, vive, N):
    var = pd.read_csv(f'{acceler}', sep=',')
    var_ = pd.read_csv(f'{vive}', sep=',')
    length = var.shape[0] // N
    length_ = var_.shape[0] // N
    for i in range(N):
        data_of_vive = var_[i * length_:length_ * (i + 1)]
        data_of_acceler = var[i * length:length * (i + 1)]
        data_of_acceler.to_csv(f'data_of_acceler_after_all_second/{i}.csv')
        data_of_vive.to_csv(f'data_of_vive_after_all_second/{i}.csv')


def Prepare_for_formulas(acc, vive, test):
    matrix_acc = []
    matrix_acc2 = []
    g = np.matrix([0, 0, 9.8])
    for file in os.listdir(acc):
        var = pd.read_csv(f'{acc}/{file}', sep=',')
        line = np.matrix([var.ax.mean() / 1000, var.ay.mean() / 1000, var.az.mean() / 1000])
        line2 = np.matrix([var.gx.mean() / 1000, var.gy.mean() / 1000, var.gz.mean() / 1000])
        matrix_acc.append(line)
        matrix_acc2.append(line2)

    # mat = np.matrix(matrix_acc)
    with open(f'for_formulas/{test}_test/acc/result_of_acc.dat', 'wb') as f:
        for line in matrix_acc:
            np.savetxt(f, line, fmt='%.8f')
        matrix_acc.clear()

    # mat = np.matrix(matrix_acc2)
    with open(f'for_formulas/{test}_test/gir/result_of_gir.txt', 'wb') as f:
        for line in matrix_acc2:
            np.savetxt(f, line, fmt='%.8f')

    for file in os.listdir(vive):
        var = pd.read_csv(f'{vive}/{file}', sep=',')
        sum = [0, 0, 0, 0]
        # line = var.Rotation.apply(lambda x:
        #                           np.array([float(x.split(',')[0][1:]), float(x.split(',')[1]),
        #                                     float(x.split(',')[2]), float(x.split(',')[3][:-1])])).mean()
        for t in range(var.shape[0]):
            line = [float(var.Rotation[t].split(',')[0][1:]), float(var.Rotation[t].split(',')[1]),
                    float(var.Rotation[t].split(',')[2]),
                    float(var.Rotation[t].split(',')[3][:-1])]
            for r in range(4):
                sum[r] += line[r]
        for t in range(4):
            sum[t] /= var.shape[0]
        matrix_acc.append(qvat_to_matrix(sum))

    with open(f'for_formulas/{test}_test/vive/result_of_vive.txt', 'wb') as f:
        for line in matrix_acc:
            np.savetxt(f, line, fmt='%.8f')

    for i in range(len(matrix_acc)):
        matrix_acc[i] = (matrix_acc[i] * g.T).T

    with open(f'for_formulas/{test}_test/vive/result.dat', 'wb') as f:
        for line in matrix_acc:
            np.savetxt(f, line, fmt='%.8f')
        matrix_acc.clear()


def qvat_to_matrix(tur):
    u = tur[0]
    v = tur[1]
    w = tur[2]
    x = tur[3]
    return np.matrix([
        [2 * (u ** 2 + v ** 2) - 1, 2 * (v * w - u * x), 2 * (v * x + u * w)],
        [2 * (v * w + u * x), 2 * (u ** 2 + w ** 2) - 1, 2 * (x * w - u * v)],
        [2 * (x * v - u * w), 2 * (w * x + u * v), 2 * (u ** 2 + x ** 2) - 1]
    ])


def Clear(folder):
    for file in os.listdir(folder):
        os.remove(f'{folder}/{file}')


# def create_matrix(name, direct):
#     for file in os.listdir(f'{name}/{direct}'):
#         if file[0] == 'q':
#             break
#         var = pd.read_csv(f'{name}/{direct}/{file}', sep=',')
#         new = pd.DataFrame()
#         matrixes = []
#         for i in range(var.shape[0]):
#             matrixes.append(qvat_to_matrix(var.Rotation[i].split(',')))
#         new['Matrix'] = matrixes
#         new.to_csv(f'{name}/{direct}/qvat_{file}')


if __name__ == '__main__':
    # Change('data_before_calibration', 'vive_before', 'acceler_before/imu_data-0-14-.csv',
    #        'acceler_before/imu_data-0-19-.csv')
    # Change('data_before_calibration', 'vive_after', 'acceler_after/imu_data-0-10-.csv',
    #        'acceler_after/imu_data-0-13-.csv')
    # Clear('data_of_acceler_after_all_first')
    # Clear('data_of_vive_after_all_first')
    # Create_data_first_test('data_after_calibration/acceler_before/full_data_of_acceler_before.csv',
    #                        'data_after_calibration/vive_before/Controller.csv', 15)
    Clear('data_of_acceler_after_all_second')
    Clear('data_of_vive_after_all_second')
    Create_data_second_test('data_after_calibration/acceler_after/full_data_of_acceler_after.csv',
                            'data_after_calibration/vive_after/Controller.csv', 50)
    Prepare_for_formulas('data_of_acceler_after_all_first', 'data_of_vive_after_all_first', 'first')
    Prepare_for_formulas('data_of_acceler_after_all_second', 'data_of_vive_after_all_second', 'second')
