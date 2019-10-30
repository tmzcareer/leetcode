""" Non-adjacent Maximum Sum """

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
val = [60, 100, 120] 
n = len(val)  
W = 50
print(dp_Knapsack(W, wt, val, n))
"""

""" 300. Longest Increasing Subsequence """

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
        if self.f[i] > 0: return self.f[r]			# 数组作递归出口
        
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














