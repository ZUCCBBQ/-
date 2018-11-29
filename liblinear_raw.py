from liblinearutil import *
import argparse
# model = train(y, x[, 'training_options'])
# y是list/tuple类型，长度为l的训练标签向量
# x是list/tuple类型的训练实例，list中的每一个元素是list/tuple/dictory类型的feature向量
#以下写法的作用是相同的
# model = train(prob[, 'training_options'])
# model = train(prob, param)




parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input", help="the full path of input file")
parser.add_argument("-in2", "--input2", help="the full path of input file")
args = parser.parse_args()

trainfile = args.input
testfile = args.input2


# examples
y, x = svm_read_problem(trainfile)
# 读入libsvm格式的数据
prob = problem(y, x)
# 将y,x写作prob
param = parameter('-s 11 -c 5  -q')

# 将参数命令写作param

# m = train(y, x, '-c 5')
# m = train(prob, '-w1 5 -c 5')
m = train(prob, param)
# 进行训练
# print(m)
# CV_ACC = train(y, x, '-v 5')
# path = '.model'
save_model(path,m)
# -v 3 是指进行3-fold的交叉验证
# 返回的是交叉验证的准确率

# best_C, best_rate = train(y, x, '-C -s 0')
testlabel, testdata = svm_read_problem(trainfile)
# param2 = parameter('')
p_labs, p_acc, p_vals = predict(testlabel, testdata, m)

print("标签\n")
print(type(p_labs))
print(p_labs)
# print(p_acc)
print("概率:\n")
print(type(p_vals))
print(p_vals)
# y是testing data的真实标签，用于计算准确率
# x是待预测样本
# p_labs: 预测出来的标签
# p_acc: tuple类型，包括准确率，MSE，Squared correlation coefficient(平方相关系数)
# p_vals: list, 直接由模型计算出来的值，没有转化成1，0的值，也可以看做是概率估计值

# (ACC, MSE, SCC) = evaluations(ty, pv)
# ty: list, 真实值
# pv: list, 估计值
