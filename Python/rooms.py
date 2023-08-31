price = int(input('enter the price'))
dis = int(input('enter the discount'))

dis_per = dis / 100
dis_pr = dis_per * price
pr_f_dis = price - dis_pr
print('the price after discount' + str(pr_f_dis))
