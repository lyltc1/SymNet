import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 查看安装包的版本
print("matplotlib version: ", matplotlib.__version__)  # 3.6.2


x = np.linspace(0, 200, 11)  # epoch, total 46.3 hours
y = np.asarray([[0,0.00911931,0.01075873,0.0107473,0.01043437646,0.009688619,0.0116866469,0.01190201879,0.01209559396,0.0121407442,0.010880689],
[0,0.00610929,0.00859152,0.010210182,0.01016606985,0.0105412839,0.01060304,0.01127199128,0.00980227308,0.0114271628,0.01159115978],
[0,0.0094613109,0.010539208,0.01093673777,0.011036898,0.0102008407,0.010769111,0.0119591053,0.0118667289,0.01173698687,0.012631688],
[0,0.00643728,0.0090264,0.0097337692,0.0106186,0.0108044,0.010617,0.010700607,0.01092376,0.01162748456,0.0112486377],
[0,0.0054055937,0.0108791322,0.01087446157,0.010927396,0.0115973844,0.01185012196,0.01182054,0.01162333,0.011685609,0.01191084],
[0,0.00578,0.00931029,0.009460272977,0.010479,0.0106798,0.01086823,0.01086875,0.010946079,0.010751985,0.010506513],
[0,0.0032996,0.00520161,0.00869842,0.0091763973,0.007944885,0.009944989,0.009338834,0.01001141,0.010272977, 0.01018371]])


y = y / np.max(y)

plt.figure(figsize=(6, 4)) 

for i, bit in enumerate(range(4, 17, 2)):
    plt.plot(x, y[i,:], label='bit' + str(bit))
# 设置x，y轴的标签
plt.xlabel('epochs')
plt.ylabel('normalized error')
plt.legend()
plt.savefig("/home/Symnet/output/tless_ablation/bit_compare.png")