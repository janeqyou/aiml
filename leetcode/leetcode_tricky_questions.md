### recursion
- calculate pow(x,n) i.e. $x^n$

```

class Solution:

    def memoization(self, x:float, n:int) -> float:
        
        cache=[1]
        for i in range(1,n+1):
            cache.append(cache[i-1]*x)
        return cache[n]

    def myPow(self, x: float, n: int) -> float:
        
        if n >= 0:
    
            return self.memoization(x,n)
            
        else:
            return self.memoization(1/x,-n)
```

The space complexity of $\mathcal{O}(n)$

-- validate bst
```
# Definition for a binary tree node.
# The left subtree of a node contains only nodes with keys less than the node's key.
# The right subtree of a node contains only nodes with keys greater than the node's key.
# Both the left and right subtrees must also be binary search trees.

# Bottom up value comparisons, not really working 

class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
        
class Solution:
    
    def validateAndValues(self, root: TreeNode) -> (bool, int, int):
        '''
        whether root is a BST
        min values in subtree of root
        max values in subtree of root 
        '''
            
        left_tree_is_valid = right_tree_is_valid = True
        left_min = left_max =  root.val - 1
        right_min = right_max = root.val + 1
        
        print(f'root value is {root.val}')
        if root.left is not None:
            left_tree_is_valid, left_min, left_max = self.validateAndValues(root.left)
            #print(f'left subtree is {left_tree_is_valid}, the min value is {left_min}, the max value is {left_max}')
        
        if root.right is not None:
            right_tree_is_valid, right_min, right_max = self.validateAndValues(root.right)
            #print(f'right subtree is {right_tree_is_valid}, the min value is {right_min}, the max value is {right_max}')
        
        
        subtree_valid = left_tree_is_valid & right_tree_is_valid 
        
        if left_max < root.val and root.val < right_min: # less than or greater than
            root_valid = True
            if root.left is None:
                left_min = left_max = root.val
            if root.right is None:
                right_min = right_max = root.val
            
        else:
            root_valid = False
            
        #print(f'left tree max is {left_max}, right tree min value is {right_min}, root is {root.val}')
        #print(f'root valid is {root_valid}')
        
        new_min = min(left_min, right_min, root.val)
        new_max = max(left_max, right_max, root.val)
            
        return (root_valid & subtree_valid, new_min, new_max)
        
   
        
    def isValidBST(self, root: TreeNode) -> bool:
        
        if root is None:
            return true
        
        else:
            
            root_is_valid,minVal,maxVal = self.validateAndValues(root)
            #print(f'root is {root_is_valid}, min value is {minVal}, max_value is {maxVal}')
            return root_is_valid
```
