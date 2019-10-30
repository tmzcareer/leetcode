<img src="http://github.com/tmzcareer/leetcode/raw/master/images/dp.jpeg" width = "300" height = "400" alt="递归方程、递归出口" align=center />


# brute force O(2^n)
# recursion 
def rec_opt(arr, i):
	if i == 0:						    # 递归出口
		return arr[0]
	elif i == 1:
		return max(arr[0], arr[1])
	else:								# 递归方程，选或不选
		A = rec_opt(arr, i-2) + arr[i]
		B = rec_opt(arr, i-1)
		return max(A, B)

arr = [1, 2, 4, 1, 7, 8, 3]
rec_opt(arr, len(arr)-1)

# dp O(n)
import numpy as np
def dp_opt(arr):
	opt = np.zeros(len(arr))			# 数组保存重叠子问题
	opt[0] = arr[0]
	opt[1] = max(arr[0], arr[1])

	for i in range(2, len(arr)):
		A = opt[i-2] + arr[i]
		B = opt[i-1]
		opt[i] = max(A,B)
	return opt[len(arr) - 1]

dp_opt(arr)



""" Subsequence Sum """

# brute force O(2^n)
# recursion 
def rec_subset(arr, i, s):
	if s == 0:
		return True
	elif i == 0:
		return arr[0] == s
	elif arr[i] > s:
		return rec_subset(arr, i-1, s)
	else:
		A = rec_subset(arr, i-1, s-arr[i])
		B = rec_subset(arr, i-1, s)
		return A or B

arr = [3, 34, 4, 12, 5, 2]
rec_subset(arr, len(arr)-1, 9)

# dp O(n)
import numpy as np
def dp_subset(arr, S):
	subset = np.zeros((len(arr), S+1), dtype=bool)
	subset[:, 0] = True
	subset[0, :] = False
	subset[0, arr[0]] = True

	for i in range(1, len(arr)):
		for s in range(1, S+1):
			if arr[i] > s:
				subset[i, s] = subset[i-1, s]
			else:
				A = subset[i-1, s-arr[i]]
				B = subset[i-1, s]
				subset[i, s] = A or B
	r,c = subset.shape
	return subset[r-1, c-1]

dp_subset(arr, 9)


""" Knapsack Problem """

# recursion
# top-down
def rec_Knapsack(wt, val, n, W): 
    if n == 0 or W == 0 : 
        return 0
    if(wt[n-1] > W): 
        return rec_Knapsack(wt, val, n-1, W)
    
    else: 
        A = val[n-1] + rec_Knapsack(wt, val, n-1, W-wt[n-1])
        B = rec_Knapsack(wt , val , n-1, W)
        return max(A,B) 

wt = [10, 20, 30]
val = [60, 100, 120] 
n = len(val)  
W = 50
print(rec_Knapsack(wt, val, n, W))

# dp
# bottom-up
"""
def dp_Knapsack(W, wt, val, n): 
    K = [[0 for x in range(W+1)] for x in range(n+1)] 
    for i in range(n+1): 
        for w in range(W+1): 
            if i==0 or w==0: 
                K[i][w] = 0
            elif wt[i-1] <= W: 
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]) 
            else: 
                K[i][w] = K[i-1][w] 
  
    return K[n][W] 
wt = [10, 20, 30]
![image](http://github.com/tmzcareer/leetcode/raw/master/images/dp.jpeg)
