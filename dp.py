""" 70. Climbing Stairs 

You are climbing a stair case. It takes n steps to reach to the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
Note: Given n will be a positive integer.

Example:
Input: 3
Output: 3

"""

# recursive
# top-down
class Solution():
    
    def climbStairs(self, n):
        self.res = [-1 for i in range(n)]       
        return self.rec(n)
        
    def rec(self, n):
        if n == 1: return 1
        elif n == 2: return 2
        elif self.res[n-1] > 0: return self.res[n-1]        

        else: 
            self.res[n-1] = self.rec(n-1) + self.rec(n-2)   
            return self.res[n-1]

# dp
# bottom-up
class Solution():
    def climbStairs(self, n):
        return self.dp(n)

    def dp(self, n):
        if n == 1: return 1                 # corner case
    
        res = [-1 for i in range(n)]
        res[0], res[1] = 1, 2
        for i in range(2, n):
            res[i] = res[i-1] + res[i-2]
        return res[n-1]




"""Problem1. Non-adjacent max-sum
https://www.youtube.com/watch?v=Jakbj4vaIbE

arr = [1, 2, 4, 1, 7, 8, 3]
n = len(arr)

"""

# brute force O(2^n)
# recursion O(n)
# top-down
res = [0 for i in range(n)]                     # 数组保存重叠子问题
def rec_opt(arr, i): 

    if i == 0: return arr[0]                     # 递归出口   
    elif i == 1: return max(arr[0], arr[1])   
    elif res[i] > 0: return res[i]

    else:                                        # 递归方程(选或不选)
        A = rec_opt(arr, i-2) + arr[i]
        B = rec_opt(arr, i-1)
        res[i] = max(A, B)
        return res[i]                       

rec_opt(arr, n-1)


# dp O(n)
# bottom-up
def dp_opt(arr):
    res = [0 for i in range(n)]
    res[0] = arr[0]
    res[1] = max(arr[0], arr[1])

    for i in range(2, n):
        A = res[i-2] + arr[i]
        B = res[i-1]
        res[i] = max(A,B)
    return res[n-1]

dp_opt(arr)










"""  Problem2. N-sum
https://www.youtube.com/watch?v=Jakbj4vaIbE

arr = [3, 34, 4, 12, 5, 2]
n = len(arr)
S = 9

"""

# brute force O(2^n)
# recursion O(n)
res = [[0 for j in range(S+1)] for i in range(n)]
def rec_subset(arr, i, s):                          # recursion比dp多一个参数index

	if s == 0: return True
	elif i == 0: return arr[0] == s
    elif res[i][s] != 0: return res[i][s]

	elif arr[i] > s:                                                     
        res[i][s] = rec_subset(arr, i-1, s)         # s必为正，否则index out of range
        return res[i][s]                            
	else:
		A = rec_subset(arr, i-1, s-arr[i])
		B = rec_subset(arr, i-1, s)
		res[i][s] = A or B
        return res[i][s]

rec_subset(arr, n-1, S)


# dp O(n)
import numpy as np
def dp_subset(arr, S):
	res = np.zeros((len(arr), S+1), dtype=bool)       # ndarray
	res[:, 0] = True
	res[0, :] = False
	res[0, arr[0]] = True

	for i in range(1, n):
		for s in range(1, S+1):
			if arr[i] > s:
				res[i, s] = res[i-1, s]
			else:
				A = res[i-1, s-arr[i]]
				B = res[i-1, s]
				res[i, s] = A or B
	return res[n-1, S+1-1]

dp_subset(arr, S)




"""Problem3. Knapsack Problem

wt = [10, 20, 30]
val = [60, 100, 120]
n = len(wt)
W = 50

"""

# recursion
# top-down
res = [0 for i in range(n+1)]
def rec_opt(n, w): 
  
    if n == 0 or w == 0 : return 0
    if res[n] > 0: return res[n]
  
    if (wt[n-1] > w):
        res[n] = rec_opt(n-1, w) 
        return res[n]   
    else: 
        A = val[n-1] + rec_opt(n-1, w-wt[n-1])
        B = rec_opt(n-1, w)
        res[n] = max(A, B)
        return res[n] 

rec_opt(n, W)

# dp
# bottom-up
def dp_opt(W): 
    res = [[0 for x in range(W+1)] for x in range(n+1)] 
    for i in range(n+1):                                             # list对比ndarray
        for w in range(W+1): 
            if i==0 or w==0: 
                res[i][w] = 0
                
            elif wt[i-1] > w:
                res[i][w] = res[i-1][w]     
            else:              
                res[i][w] = max(val[i-1] + res[i-1][w-wt[i-1]], res[i-1][w]) 
    return res[n][w]

dp_opt(W)










""" 300. Longest Increasing Subsequence
Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:
Input: [10,9,2,5,3,7,101,18]
Output: 4

"""

# brute force O(2^n)
# recursion + memorization O(n^2)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
            
        n = len(nums)
        if n == 0: return 0
        self.f = [0 for i in range(n)]				# 数组初始化
        ans = 0
        for i in range(n):
            ans = max(ans, self.LIS(nums,i))
        return ans
    
    
    def LIS(self, nums, i):
        if i == 0: return 1
        if self.f[i] > 0: return self.f[i]			# 数组作递归出口
        
        ans = 1
        for j in range(i):
            if nums[j] < nums[i]:
                ans = max(ans, self.LIS(nums, j) + 1)
        self.f[i] = ans;							# 数组保存重叠子问题
        return self.f[i]
        
# dp O(n^2)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
            
        n = len(nums)
        if n == 0: return 0
        
        self.f = [1 for i in range(n)]
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    self.f[i] = max(self.f[i], self.f[j]+1)
        return max(self.f)

""" O(nlogn) 想一想




"""




# 416. Partition Equal Subset Sum

# 回溯
# 0/1 背包 记忆递归
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = 0
        for num in nums:
            total += num
        if total % 2 != 0: return False
        total /= 2
        return self.helper(nums, 0, total)
    
    def helper(self, nums, index, target):
        if target == 0: return True
        if index == len(nums) or target < 0: return False
        if self.helper(nums, index + 1, target - nums[index]) : return True
        # 1，1，1，1000
        j = index + 1
        while j < len(nums) and nums[index] == nums[j]:
            j += 1
        return self.helper(nums, j, target)

# dp
import numpy as np
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = 0
        for num in nums:
            total += num
        if total % 2 != 0: return False
        total = int(total/2)    # python数据类型返回float
        dp = np.zeros(total+1, dtype=bool)  #只能初始化即填充 无法像java一样
        dp[0] = True
        
        for num in nums:
            for j in range(total, 0, -1):
                if j >= num:
                    dp[j] = dp[j] or dp[j-num] # 选或不选
        return dp[total]





# 474. Ones and Zeroes

#dp

class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        # m个0, n个1能组成的最长字符串
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for str in strs:
            zeros, ones = 0, 0
            for c in str:
                if c == "0":
                    zeros += 1
                elif c == "1":
                    ones += 1
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        return dp[m][n]





""" 688 Knight Probability in Chessboard

On an NxN chessboard, a knight starts at the r-th row and c-th column and attempts to make exactly K moves. 
Return the probability that the knight remains on the board after it has stopped moving.

Example:
Input:  N=3, K=2, r=0, c=0
Output: 0.0625

"""

# Solution 1: dynamic programming 

class Solution:
    def knightProbability(self, N: int, K: int, r: int, c: int) -> float:
        dp0=[[0 for i in range(N)] for j in range(N)]
        dp0[r][c]=1.0
        dirs=[[1,2],[-1,-2],[1,-2],[-1,2],
              [2,1],[-2,-1],[2,-1],[-2,1]]
        
        for k in range(K):
            dp1=[[0 for i in range(N)] for j in range(N)]
            for i in range(N):
                for j in range(N):
                    for m in range(8):
                        x=j+dirs[m][0]
                        y=i+dirs[m][1]
                        if x<0 or y<0 or x>=N or y>=N:
                            continue
                        dp1[y][x]+=1/8*dp0[i][j]
            dp0,dp1=dp1,dp0
        total=0
        for i in range(N):
            for j in range(N):
                total+=dp0[i][j]
                
        return total

# Solution 2: dfs

class Solution(object):
    def knightProbability(self, N, K, r, c):
        
        moves = [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]]
        mem = {}
        
        def dfs(N,K,r,c):
            if r>N-1 or c>N-1 or r<0 or c<0:
                return 0
            if K==0:
                return 1
            if (K,r,c) in mem:
                return mem[(K,r,c)]
            rate = 0.0
            for i in range(len(moves)):
                rate += 0.125*dfs(N,K-1,r+moves[i][0],c+moves[i][1])
            mem[(K,r,c)] = rate   # TLE
            return rate
        
        ans = dfs(N,K,r,c)
        return ans











